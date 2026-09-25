"""Evaluation of the LLM cascade and its baselines on held-out images."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import serving
from .data import load_pil, location_from_mask


def eval_items(splits, masks_by_category):
    items = []
    for cat, s in splits.items():
        items += [{"category": cat, "path": p, "label": 0, "true_type": "none", "true_location": "none"} for p in s["eval_good"]]
        items += [{"category": cat, "path": d["path"], "label": 1, "true_type": d["type"],
                   "true_location": location_from_mask(masks_by_category[cat][d["mask"]])} for d in s["defect_eval"]]
    return items


def val_good_items(splits):
    return [{"category": c, "path": p, "label": 0, "true_type": "none", "true_location": "none"}
            for c, s in splits.items() for p in s["val_good"]]


def run_config(name, tier, items, splits, model=None, with_reference=False, workers=16, log=print):
    """Ask one model configuration about every item. Returns a DataFrame with answers and timing."""
    refs = {}
    def job(it):
        imgs = [load_pil(it["path"])]
        if with_reference:
            if it["category"] not in refs:
                refs[it["category"]] = serving.reference_image(splits, it["category"])
            imgs = [refs[it["category"]]] + imgs
        return tier.safe_ask(imgs, it["category"], model)
    t0 = time.perf_counter()
    log(f"{name}: {len(items)} images")
    res = serving.run_many(job, [(it,) for it in items], workers=workers, log=log)
    wall = time.perf_counter() - t0
    df = pd.DataFrame([{**it, **{k: r.get(k) for k in ("verdict", "defect_type", "location", "explanation", "valid",
                                                        "p_defective", "latency_s", "prompt_tokens", "completion_tokens", "raw")}}
                       for it, r in zip(items, res)])
    df["config"] = name; df.attrs["wall_s"] = wall
    return df


def summarize(df, name=None, wall_s=None):
    y = df.label.values; pred = (df.verdict == "defective").values
    det = df[(df.label == 1) & (df.verdict == "defective")]
    s = df.p_defective.astype(float).values
    return {"config": name or df.config.iloc[0], "images": len(df),
            "roc_auc": roc_auc_score(y, s) if len(set(y)) > 1 else np.nan,
            "accuracy": (pred == (y == 1)).mean(), "recall": pred[y == 1].mean(), "false_positive_rate": pred[y == 0].mean(),
            "defect_type_acc": (det.defect_type == det.true_type).mean() if len(det) and det.defect_type.notna().any() else np.nan,
            "location_acc": (det.location == det.true_location).mean() if len(det) and det.location.notna().any() else np.nan,
            "valid_json_rate": df.valid.mean(),
            "mean_latency_s": df.latency_s.dropna().mean(),
            "throughput_img_per_s": (len(df) / wall_s) if wall_s else np.nan}


def resnet_baseline(vision_preds, items):
    """The classical (non-LLM) baseline: ResNet-18 recipe C scores on the same images."""
    vp = vision_preds[vision_preds.variant == "C_real_defects"].set_index("path")
    df = pd.DataFrame(items)
    df["p_defective"] = df.path.map(vp.score)
    df["verdict"] = np.where(df.p_defective >= 0.5, "defective", "good")
    df["valid"] = True; df["defect_type"] = None; df["location"] = None; df["latency_s"] = np.nan
    df["config"] = "Baseline: ResNet-18 classifier (non-LLM)"
    return df


# ---------------------------------------------------------------- the per-part cascade (shared by notebook, stress tests and app)
DECISION_OF = {"accept": "accept", "reject": "reject", "escalate_to_human": "manual_review"}


class InputCheck:
    """Grey-level statistics of each product's good training images. Flags broken or foreign frames
    (lens cap, overexposure, wrong product) before any model is asked. No ML, microseconds."""

    def __init__(self, splits, n=60, limit=40.0):
        self.stats, self.limit = {}, limit
        for cat, s in splits.items():
            st = np.array([self._stats(load_pil(p)) for p in s["fit_good"][:n]])
            self.stats[cat] = (st.mean(0), st.std(0) + 1e-6)

    @staticmethod
    def _stats(pil):
        g = np.asarray(pil.convert("L").resize((256, 256)), dtype=np.float32)
        return g.mean(), g.std()

    def __call__(self, pil, cat):
        mu, sd = self.stats[cat]
        z = float(np.abs((np.array(self._stats(pil)) - mu) / sd).max())
        return {"out_of_distribution": z > self.limit, "z": round(z, 1), "z_limit": self.limit}


def decide_part(pil, cat, *, t1, t2, refs, check, policy_fn, t_lo, force_t2=False, audit=False, explainer=None):
    """Run one part through the cascade. Returns decision, tier reached, evidence bucket, reason, and both tiers' answers.
    Never raises: model failures fall back to escalation."""
    t0 = time.perf_counter()
    try:
        pil = pil.convert("RGB"); pil.load()
    except Exception as exc:
        return {"decision": "manual_review", "tier": 3, "bucket": "unreadable", "reason": f"unreadable image ({exc})",
                "r1": None, "r2": None, "input_check": None, "total_s": time.perf_counter() - t0}
    chk = check(pil, cat)
    if chk["out_of_distribution"]:
        return {"decision": "manual_review", "tier": 3, "bucket": "input_ood", "input_check": chk, "r1": None, "r2": None,
                "reason": f"image does not look like a {cat} from training (z={chk['z']}); check the camera and the part",
                "total_s": time.perf_counter() - t0}
    r1 = t1.safe_ask([pil], cat); r2 = None
    p1 = float(r1["p_defective"]) if r1.get("valid") else 1.0          # tier-1 failure: treat as flagged
    if p1 < t_lo and not force_t2 and not audit:
        return {"decision": "accept", "tier": 1, "bucket": "fast_accept", "input_check": chk, "r1": r1, "r2": None,
                "reason": f"tier 1 is confident the part is good (P(defect) {p1:.1%})", "total_s": time.perf_counter() - t0}
    if explainer is not None:
        # The fine-tuned tier 2 decides; the untrained base model (same server) writes the operator's sentence.
        # Both requests run at the same time, so this adds almost no waiting.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            f2 = ex.submit(t2.safe_ask, [refs[cat], pil], cat); fe = ex.submit(t2.safe_ask, [refs[cat], pil], cat, explainer)
            r2, re_ = f2.result(), fe.result()
        if r2.get("valid") and re_.get("explanation") and re_.get("verdict") == r2.get("verdict"):   # never contradict the decision
            r2 = {**r2, "explanation_fine_tuned": r2.get("explanation"), "explanation": re_["explanation"], "explanation_by": explainer}
    else:
        r2 = t2.safe_ask([refs[cat], pil], cat)
    action, reason, bucket = policy_fn(p1, t_lo, r2["verdict"] if r2.get("valid") else None)
    if audit and p1 < t_lo:
        reason = "random audit of a tier-1 accept: " + reason
    return {"decision": DECISION_OF[action], "tier": 3 if action == "escalate_to_human" else 2, "bucket": bucket,
            "reason": reason, "input_check": chk, "r1": r1, "r2": r2, "total_s": time.perf_counter() - t0}
