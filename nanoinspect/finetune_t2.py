"""LoRA fine-tuning of the tier-2 model (Qwen3.8-27B, BF16) on the ZGX Nano.

Same training data as tier 1 (split-A real defects + an equal number of good training images; nothing
from the evaluation split), but in tier 2's exact input format: a known-good reference image, the part,
and the tier-2 prompt. The target is the JSON verdict, defect type, location and a short explanation.
The NVFP4 serving build cannot be trained, so we train on the BF16 weights and serve BF16 + the
adapter with vLLM.
    python -m nanoinspect.finetune_t2 [--limit N] [--epochs 1]
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch

from . import data, serving
from .config import CATEGORIES, MODELS_DIR, SEED
from .data import load_pil, location_from_mask

BASE = "Qwen/Qwen3.8-27B"
ADAPTER_DIR = MODELS_DIR / "vlm_lora_t2"


def explanation(label, dtype, loc):
    if label == 0:
        return "No defect: differences from the reference are normal variation in position, lighting and texture."
    return f"Visible {dtype.replace('_', ' ')} at the {loc} of the part that the reference part does not have."


def examples(splits, masks, seed=SEED):
    rng = random.Random(seed); ex = []
    for cat, s in splits.items():
        for d in s["defect_fit"]:
            loc = location_from_mask(masks[cat][d["mask"]])
            ex.append({"path": d["path"], "category": cat, "answer": json.dumps(
                {"verdict": "defective", "defect_type": d["type"], "location": loc, "explanation": explanation(1, d["type"], loc)})})
        for p in rng.sample(s["fit_good"][1:], len(s["defect_fit"])):     # fit_good[0] is the reference image
            ex.append({"path": p, "category": cat, "answer": json.dumps(
                {"verdict": "good", "defect_type": "none", "location": "none", "explanation": explanation(0, None, None)})})
    rng.shuffle(ex)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    splits = data.make_splits(); masks = {c: data.load_cache(c, splits)[1] for c in CATEGORIES}
    ex = examples(splits, masks)
    if a.limit: ex = ex[:a.limit]
    refs = {c: serving.reference_image(splits, c).convert("RGB").resize((serving.IMG_SIZE, serving.IMG_SIZE)) for c in CATEGORIES}

    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(BASE)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    print(f"loaded {BASE} in {time.perf_counter() - t0:.0f}s, {model.get_memory_footprint() / 1e9:.1f} GB", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    pm = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad); total = sum(p.numel() for p in pm.parameters())
    print(f"trainable {trainable / 1e6:.1f}M of {total / 1e9:.1f}B ({100 * trainable / total:.3f}%)", flush=True)

    def prompt_text(cat):
        msgs = [{"role": "system", "content": [{"type": "text", "text": serving.SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": serving.tier2_prompt(cat)}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    eos = proc.tokenizer.eos_token or "<|im_end|>"
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.max_steps or math.ceil(len(ex) * a.epochs / (a.batch * a.accum)); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    tok = proc.tokenizer; tok.padding_side = "right"
    pm.train(); step = micro = 0; hist = []; run = []; t0 = time.perf_counter()
    order = (ex * math.ceil(a.epochs))[: int(len(ex) * a.epochs)]
    for i in range(0, len(order), a.batch):
        if step >= steps: break
        b = order[i:i + a.batch]
        imgs = [im for e in b for im in (refs[e["category"]], load_pil(e["path"]).convert("RGB").resize((serving.IMG_SIZE, serving.IMG_SIZE)))]
        ptxt = [prompt_text(e["category"]) for e in b]
        full = proc(text=[p + e["answer"] + eos for p, e in zip(ptxt, b)], images=imgs, return_tensors="pt", padding=True)
        pr = proc(text=ptxt, images=imgs, return_tensors="pt", padding=True)
        labels = full["input_ids"].clone()
        for j, n in enumerate(pr["attention_mask"].sum(1).tolist()):
            labels[j, :n] = -100
        labels[full["attention_mask"] == 0] = -100
        loss = pm(**full.to(pm.device), labels=labels.to(pm.device)).loss
        (loss / a.accum).backward(); run.append(loss.item()); micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            el = time.perf_counter() - t0
            hist.append({"step": step, "loss": float(np.mean(run)), "elapsed_s": el}); run = []
            print(f"step {step}/{steps} loss {hist[-1]['loss']:.4f} {el / step:.1f}s/step eta {(steps - step) * el / step / 60:.0f} min", flush=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True); pm.save_pretrained(str(ADAPTER_DIR))
    stats = {"base": BASE, "examples": len(ex), "epochs": a.epochs, "optimizer_steps": step, "train_seconds": round(time.perf_counter() - t0, 1),
             "trainable_params": trainable, "total_params": total, "trainable_pct": round(100 * trainable / total, 3), "lora_rank": a.rank,
             "lr": a.lr, "effective_batch": a.batch * a.accum, "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    (ADAPTER_DIR / "training_stats.json").write_text(json.dumps({"stats": stats, "history": hist}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
