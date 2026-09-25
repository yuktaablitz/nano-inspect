"""Qwen2.5-VL as a visual inspector: zero-shot inference, LoRA fine-tuning, evaluation."""
import json
import math
import random
import re
import threading
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import LOCATIONS, MODELS_DIR, SEED, VLM_ID, VLM_IMAGE_SIZE
from .data import defect_types, load_pil, location_from_mask

ADAPTER_DIR = MODELS_DIR / "vlm_lora"
SYSTEM = ("You are an industrial visual quality inspector. You look at one product image and report "
          "whether the product has a manufacturing defect.")


def prompt_text(category):
    return (f"Product: {category.replace('_', ' ')}.\n"
            f"Possible defect types for this product: {', '.join(defect_types(category))}.\n"
            f"Image regions: {', '.join(LOCATIONS)}.\n"
            "Inspect the product. Answer with JSON only, in exactly this form: "
            '{"verdict": "good" or "defective", "defect_type": one of the defect types or "none", '
            '"location": one of the image regions or "none"}')


def messages(category, answer=None):
    m = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(category)}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def answer_json(label, defect_type="none", location="none"):
    if label == 0:
        return json.dumps({"verdict": "good", "defect_type": "none", "location": "none"})
    return json.dumps({"verdict": "defective", "defect_type": defect_type, "location": location})


def prep_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("RGB").resize((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE), Image.BICUBIC)


_GOOD = {"good", "ok", "normal", "pass", "no defect", "non-defective", "acceptable"}
_BAD = {"defective", "defect", "bad", "anomalous", "fail", "damaged", "faulty"}


def parse_answer(text, category):
    m = re.search(r"\{.*?\}", str(text), re.S)
    out = {"verdict": None, "defect_type": None, "location": None, "valid": False}
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    v = str(d.get("verdict", "")).strip().lower()
    out["verdict"] = "good" if v in _GOOD else "defective" if v in _BAD else None
    t = str(d.get("defect_type", "none")).strip().lower().replace(" ", "_")
    out["defect_type"] = t if t in set(defect_types(category)) | {"none"} else f"other:{t}"
    loc = str(d.get("location", "none")).strip().lower().replace(" ", "-")
    out["location"] = loc if loc in set(LOCATIONS) | {"none"} else f"other:{loc}"
    out["valid"] = out["verdict"] is not None
    return out


class VLM:
    """One copy of the base model in memory. A LoRA adapter can be attached; zero_shot()
    temporarily disables it, so both variants are compared with the same weights."""

    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        t0 = time.perf_counter()
        try:
            self.processor = AutoProcessor.from_pretrained(VLM_ID, local_files_only=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VLM_ID, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
        except OSError:
            self.processor = AutoProcessor.from_pretrained(VLM_ID)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_ID, dtype=torch.bfloat16, device_map={"": 0})
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0
        self.adapter_loaded = False
        self.lock = threading.Lock()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def attach_adapter(self, path=ADAPTER_DIR):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, str(path)); self.model.eval()
        self.adapter_loaded = True

    @contextmanager
    def zero_shot(self):
        with (self.model.disable_adapter() if self.adapter_loaded else nullcontext()):
            yield

    def _chat(self, category, answer=None, generation=False):
        return self.processor.apply_chat_template(messages(category, answer), tokenize=False,
                                                  add_generation_prompt=generation)

    @torch.inference_mode()
    def predict(self, images, categories, batch_size=8, max_new_tokens=48):
        """images: PIL images or arrays. Returns one dict per image (parsed answer + raw text + latency)."""
        results = []
        tok = self.processor.tokenizer
        for i in range(0, len(images), batch_size):
            imgs = [prep_image(im) for im in images[i:i + batch_size]]
            cats = categories[i:i + batch_size]
            with self.lock:
                tok.padding_side = "left"
                inputs = self.processor(text=[self._chat(c, generation=True) for c in cats], images=imgs,
                                        return_tensors="pt", padding=True).to(self.device)
                t0 = time.perf_counter()
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                dt = time.perf_counter() - t0
            texts = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for c, t in zip(cats, texts):
                results.append({**parse_answer(t, c), "raw": t, "latency_s": dt / len(imgs), "batch_size": len(imgs)})
        return results


# ---------------------------------------------------------------- fine-tuning
def finetune_examples(splits, masks_by_category, seed=SEED):
    """Split-A real defects (label, type, location from the ground-truth mask) plus an equal
    number of good training images per category."""
    rng = random.Random(seed); ex = []
    for cat, s in splits.items():
        for d in s["defect_fit"]:
            loc = location_from_mask(masks_by_category[cat][d["mask"]])
            ex.append({"path": d["path"], "category": cat, "answer": answer_json(1, d["type"], loc)})
        for p in rng.sample(s["fit_good"], len(s["defect_fit"])):
            ex.append({"path": p, "category": cat, "answer": answer_json(0)})
    rng.shuffle(ex)
    return ex


def finetune(vlm, examples, out_dir=ADAPTER_DIR, epochs=2, batch_size=4, grad_accum=2, lr=1e-4, rank=16,
             gradient_checkpointing=False, log_every=10, log=print):
    """LoRA on the language model's attention and MLP projections; the vision encoder stays frozen.
    Gradient checkpointing is off by default: the Nano's 128 GB unified memory makes the speed worth more."""
    from peft import LoraConfig, get_peft_model
    model = vlm.model
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    pm = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pm.parameters())
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = math.ceil(len(examples) * epochs / (batch_size * grad_accum)); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    tok = vlm.processor.tokenizer
    rng = random.Random(SEED); history = []; step = 0; micro = 0; t0 = time.perf_counter(); running = []
    pm.train()
    for epoch in range(epochs):
        order = examples[:]; rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            imgs = [prep_image(load_pil(e["path"])) for e in batch]
            tok.padding_side = "right"
            full = vlm.processor(text=[vlm._chat(e["category"], e["answer"]) for e in batch], images=imgs,
                                 return_tensors="pt", padding=True)
            prompt = vlm.processor(text=[vlm._chat(e["category"], generation=True) for e in batch], images=imgs,
                                   return_tensors="pt", padding=True)
            labels = full["input_ids"].clone()
            for j, n in enumerate(prompt["attention_mask"].sum(1).tolist()):
                labels[j, :n] = -100
            labels[full["attention_mask"] == 0] = -100
            loss = pm(**full.to(vlm.device), labels=labels.to(vlm.device)).loss
            (loss / grad_accum).backward(); running.append(loss.item()); micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % log_every == 0 or step == steps:
                    history.append({"step": step, "epoch": epoch + 1, "loss": float(np.mean(running)),
                                    "lr": sched.get_last_lr()[0], "elapsed_s": time.perf_counter() - t0})
                    log(f"step {step}/{steps}  epoch {epoch + 1}  loss {np.mean(running):.4f}  "
                        f"{(time.perf_counter() - t0) / step:.1f}s/step"); running = []
    if micro % grad_accum:
        opt.step(); opt.zero_grad(set_to_none=True)
    train_s = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True); pm.save_pretrained(str(out_dir))
    if gradient_checkpointing:
        model.gradient_checkpointing_disable()
    pm.eval(); vlm.model = pm; vlm.adapter_loaded = True
    stats = {"examples": len(examples), "epochs": epochs, "optimizer_steps": step, "train_seconds": round(train_s, 1),
             "trainable_params": trainable, "total_params": total, "trainable_pct": round(100 * trainable / total, 3),
             "lora_rank": rank, "lr": lr, "effective_batch": batch_size * grad_accum,
             "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    (out_dir / "training_stats.json").write_text(json.dumps({"stats": stats, "history": history}, indent=1))
    return stats, pd.DataFrame(history)


# ---------------------------------------------------------------- evaluation
def eval_items(splits, masks_by_category):
    items = []
    for cat, s in splits.items():
        items += [{"category": cat, "path": p, "label": 0, "true_type": "none", "true_location": "none"} for p in s["eval_good"]]
        items += [{"category": cat, "path": d["path"], "label": 1, "true_type": d["type"],
                   "true_location": location_from_mask(masks_by_category[cat][d["mask"]])} for d in s["defect_eval"]]
    return items


def evaluate(vlm, items, mode, batch_size=8, log=print):
    """mode: 'zero_shot' (adapter disabled) or 'fine_tuned'."""
    rows = []; t0 = time.perf_counter()
    ctx = vlm.zero_shot() if mode == "zero_shot" else nullcontext()
    with ctx:
        chunk = 64
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            preds = vlm.predict([load_pil(it["path"]) for it in part], [it["category"] for it in part], batch_size)
            rows += [{**it, **p, "mode": mode} for it, p in zip(part, preds)]
            log(f"\r{mode}: {len(rows)}/{len(items)} images, {time.perf_counter() - t0:.0f}s", end="")
    log("")
    return pd.DataFrame(rows)


def summarize(df):
    """Detection metrics + defect-type and location accuracy (on correctly detected defects)."""
    def one(g):
        y = g.label.values; pred = (g.verdict == "defective").values
        det = g[(g.label == 1) & (g.verdict == "defective")]
        return pd.Series({
            "images": len(g), "accuracy": ((pred == (y == 1))).mean(),
            "recall": pred[y == 1].mean() if (y == 1).any() else np.nan,
            "false_positive_rate": pred[y == 0].mean() if (y == 0).any() else np.nan,
            "valid_json_rate": g.valid.mean(),
            "defect_type_acc": (det.defect_type == det.true_type).mean() if len(det) and det.defect_type.notna().any() else np.nan,
            "location_acc": (det.location == det.true_location).mean() if len(det) and det.location.notna().any() else np.nan,
            "s_per_image": g.latency_s.mean()})
    return df.groupby("mode").apply(one, include_groups=False)
