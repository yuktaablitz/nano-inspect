"""Clients for the two vision-language models served locally by vLLM on the ZGX Nano.

Tier 1 (cheap):     Qwen2.5-VL-7B + NanoInspect LoRA (fine-tuned on the Nano)   http://127.0.0.1:8001
Tier 2 (expensive): Qwen3.8-27B + NanoInspect LoRA (BF16), or the untrained NVFP4 build (vLLM recipes for DGX Spark / GB10)  http://127.0.0.1:8002

Both speak the OpenAI-compatible API. Each answer is JSON; we also read the log-probabilities of the
verdict token, which turns the model's answer into a calibrated-looking score P(defective) that can be
thresholded and scored with ROC-AUC like any classifier.
"""
import base64
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

from .config import LOCATIONS
from .data import defect_types, load_pil

TIER1_URL = os.environ.get("NANOINSPECT_TIER1_URL", "http://127.0.0.1:8001")
TIER2_URL = os.environ.get("NANOINSPECT_TIER2_URL", "http://127.0.0.1:8002")
TIER1_MODEL = os.environ.get("NANOINSPECT_TIER1_MODEL", "nanoinspect-7b-lora")
TIER1_BASE = "qwen2.5-vl-7b"
TIER2_MODEL = os.environ.get("NANOINSPECT_TIER2_MODEL", "qwen3.8-27b")        # adapter name when the fine-tuned tier 2 is served
TIER2_BASE = os.environ.get("NANOINSPECT_TIER2_BASE", "qwen3.8-27b")          # the untrained 27B, for the zero-shot baseline
IMG_SIZE = 448

SYSTEM = ("You are an industrial visual quality inspector. You look at one product image and report "
          "whether the product has a manufacturing defect.")

_GOOD = {"good", "ok", "normal", "pass", "no defect", "non-defective", "acceptable"}
_BAD = {"defective", "defect", "bad", "anomalous", "fail", "damaged", "faulty"}


def to_data_url(img, size=IMG_SIZE, fmt="PNG"):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO(); img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def tier1_prompt(category):
    """Identical to the fine-tuning prompt (nanoinspect/vlm.py)."""
    return (f"Product: {category.replace('_', ' ')}.\n"
            f"Possible defect types for this product: {', '.join(defect_types(category))}.\n"
            f"Image regions: {', '.join(LOCATIONS)}.\n"
            "Inspect the product. Answer with JSON only, in exactly this form: "
            '{"verdict": "good" or "defective", "defect_type": one of the defect types or "none", '
            '"location": one of the image regions or "none"}')


def tier2_prompt(category):
    """Tier 2 confirms or clears tier 1's flags, so it is asked to be precise: normal part-to-part variation
    is not a defect. (A first, less conservative prompt flagged 39% of good parts; this one 6%, on a 120-image check.)"""
    return (f"Product: {category.replace('_', ' ')}.\n"
            "The FIRST image is a known-good reference part. The SECOND image is the part to inspect.\n"
            "Good parts are never pixel-identical: position, rotation, lighting, focus, natural texture and small colour "
            "variation are normal and are NOT defects. Answer \"defective\" only if the second part has a clear manufacturing "
            "defect (crack, break, hole, scratch, cut, contamination, missing or extra material, deformation, misprint, wrong part) "
            "that a quality inspector would reject. If you are not sure, answer \"good\".\n"
            f"Possible defect types for this product: {', '.join(defect_types(category))}.\n"
            f"Image regions: {', '.join(LOCATIONS)}.\n"
            "Answer with JSON only: "
            '{"verdict": "good" or "defective", "defect_type": one of the defect types or "none", '
            '"location": one of the image regions or "none", "explanation": one short sentence for the operator}')


def parse(text, category):
    out = {"verdict": None, "defect_type": None, "location": None, "explanation": None, "valid": False}
    m = re.search(r"\{.*?\}", str(text), re.S)
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
    out["explanation"] = d.get("explanation")
    out["valid"] = out["verdict"] is not None
    return out


def p_defective(logprobs, verdict):
    """P(defective) from the top-k alternatives at the verdict token. Falls back to 0/1 from the verdict."""
    if logprobs:
        for tok in logprobs:
            t = tok["token"].strip().strip('"').lower()
            if t and ("good".startswith(t) or "defective".startswith(t) or t.startswith(("good", "def"))) and len(t) >= 2:
                g = d = 0.0
                for alt in tok.get("top_logprobs", []):
                    a = alt["token"].strip().strip('"').lower()
                    if len(a) >= 2 and ("good".startswith(a) or a.startswith("good")): g += math.exp(alt["logprob"])
                    elif len(a) >= 2 and ("defective".startswith(a) or a.startswith("def")): d += math.exp(alt["logprob"])
                if g + d > 0:
                    return d / (g + d)
                break
    return {"defective": 1.0, "good": 0.0}.get(verdict, 0.5)


class Tier:
    def __init__(self, name, url, model, prompt_fn, n_images=1, extra=None, max_tokens=48, timeout=180):
        self.name, self.url, self.model, self.prompt_fn = name, url.rstrip("/"), model, prompt_fn
        self.n_images, self.extra, self.max_tokens, self.timeout = n_images, extra or {}, max_tokens, timeout
        self.session = requests.Session()

    def messages(self, images, category):
        content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}} for im in images]
        content.append({"type": "text", "text": self.prompt_fn(category)})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def ask(self, images, category, model=None):
        """images: list (length n_images) of PIL/arrays. Returns parsed answer + p_defective + timing."""
        body = {"model": model or self.model, "messages": self.messages(images, category), "temperature": 0,
                "max_tokens": self.max_tokens, "logprobs": True, "top_logprobs": 5, **self.extra}
        t0 = time.perf_counter()
        r = self.session.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json(); ch = j["choices"][0]
        text = ch["message"].get("content") or ""
        p = parse(text, category)
        lp = (ch.get("logprobs") or {}).get("content")
        return {**p, "p_defective": p_defective(lp, p["verdict"]), "raw": text, "latency_s": dt,
                "prompt_tokens": j.get("usage", {}).get("prompt_tokens"), "completion_tokens": j.get("usage", {}).get("completion_tokens")}

    def safe_ask(self, images, category, model=None):
        try:
            return self.ask(images, category, model)
        except Exception as exc:
            return {"verdict": None, "valid": False, "p_defective": 0.5, "raw": f"error: {exc}", "latency_s": None,
                    "defect_type": None, "location": None, "explanation": None, "error": repr(exc)}

    def healthy(self):
        try:
            return self.session.get(f"{self.url}/v1/models", timeout=3).ok
        except Exception:
            return False

    def metrics(self):
        """A few headline numbers from vLLM's Prometheus endpoint."""
        try:
            txt = self.session.get(f"{self.url}/metrics", timeout=3).text
        except Exception:
            return {}
        def val(name, agg=sum):
            xs = [float(l.rsplit(" ", 1)[1]) for l in txt.splitlines() if l.startswith(name) and not l.startswith("#")]
            return agg(xs) if xs else None
        def hist_mean(base):
            s, c = val(base + "_sum"), val(base + "_count")
            return (s / c) if s and c else None
        return {"requests_running": val("vllm:num_requests_running"), "requests_waiting": val("vllm:num_requests_waiting"),
                "kv_cache_usage": val("vllm:kv_cache_usage_perc"),
                "prefix_cache_hit_rate": (val("vllm:prefix_cache_hits_total") / val("vllm:prefix_cache_queries_total"))
                                         if val("vllm:prefix_cache_queries_total") else None,
                "prompt_tokens_total": val("vllm:prompt_tokens_total"), "generation_tokens_total": val("vllm:generation_tokens_total"),
                "requests_finished": val("vllm:request_success_total"),
                "mean_ttft_s": hist_mean("vllm:time_to_first_token_seconds"),
                "mean_e2e_s": hist_mean("vllm:e2e_request_latency_seconds"),
                "mean_tpot_s": hist_mean("vllm:time_per_output_token_seconds") or hist_mean("vllm:inter_token_latency_seconds")}


def tier1():
    return Tier("tier1", TIER1_URL, TIER1_MODEL, tier1_prompt, n_images=1)


T2_ADAPTER = "nanoinspect-27b-lora"


def tier2():
    """Uses the fine-tuned tier-2 adapter whenever the server offers it (BF16 + LoRA), else the base model (NVFP4)."""
    model = TIER2_MODEL
    if "NANOINSPECT_TIER2_MODEL" not in os.environ:
        try:
            ids = [m["id"] for m in requests.get(f"{TIER2_URL}/v1/models", timeout=3).json()["data"]]
            if T2_ADAPTER in ids: model = T2_ADAPTER
        except Exception:
            pass
    return Tier("tier2", TIER2_URL, model, tier2_prompt, n_images=2, max_tokens=120,
                extra={"chat_template_kwargs": {"enable_thinking": False}})


def reference_image(splits, category, k=0):
    """A known-good reference part from the training split (never an evaluation image)."""
    return load_pil(splits[category]["fit_good"][k])


def run_many(fn, jobs, workers=16, log=None):
    """Run fn(*job) concurrently, preserving order."""
    out = [None] * len(jobs); t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
        for n, f in enumerate(__import__("concurrent.futures").futures.as_completed(futs), 1):
            out[futs[f]] = f.result()
            if log and (n % 50 == 0 or n == len(jobs)):
                log(f"\r{n}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", end="")
    if log: log("")
    return out


def load_test(tier, jobs, concurrency, model=None):
    """Serving benchmark at a fixed number of concurrent clients."""
    lat, toks = [], 0
    t0 = time.perf_counter()
    res = run_many(lambda ims, c: tier.ask(ims, c, model), jobs, workers=concurrency)
    wall = time.perf_counter() - t0
    lat = [r["latency_s"] for r in res]; toks = sum(r["completion_tokens"] or 0 for r in res)
    return {"concurrency": concurrency, "requests": len(jobs), "wall_s": round(wall, 2),
            "requests_per_s": round(len(jobs) / wall, 2), "output_tokens_per_s": round(toks / wall, 1),
            "p50_latency_s": round(float(np.percentile(lat, 50)), 3), "p95_latency_s": round(float(np.percentile(lat, 95)), 3)}
