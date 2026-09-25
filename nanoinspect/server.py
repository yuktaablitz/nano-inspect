"""NanoInspect Edge: the web application that runs on the HP ZGX Nano.

    ./serve_models.sh both        # tier 1 (:8001) and tier 2 (:8002) on vLLM
    python -m nanoinspect.server  # http://127.0.0.1:8080

Env: NANOINSPECT_CLOUD_URL (default http://127.0.0.1:9000), NANOINSPECT_CLOUD_KEY, NANOINSPECT_SITE,
     NANOINSPECT_HOST / NANOINSPECT_PORT
"""
import base64
import collections
import io
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from matplotlib import cm as _cm
plt_cm = lambda x: _cm.jet(x)[..., :3]
from pydantic import BaseModel

from . import data, policy, serving, telemetry
from .actions import ActionBuilder
from .delta import DeltaMap
from .cascade import InputCheck, decide_part, eval_items
from .config import CATEGORIES, DATA_ROOT, LOCATIONS, RESULTS_DIR
from .escalation import CloudSync, EdgeStore
from .linesim import DECISION_OF, LineSimulator

WEB = Path(__file__).parent / "web"
DOCS = Path(__file__).resolve().parent.parent / "docs"
SITE = os.environ.get("NANOINSPECT_SITE", "Plant-SJ / Line 3 / QC-2")
CLOUD_URL = os.environ.get("NANOINSPECT_CLOUD_URL", "http://127.0.0.1:9000")
CLOUD_KEY = os.environ.get("NANOINSPECT_CLOUD_KEY", "nanoinspect-demo-key")
SOP = {"accept": "Release the {cat} to the next station.",
       "manual_review": "Hold the {cat} until the reviewer's decision arrives.",
       "reject": "Reject the {cat}, tag it with the defect type, and re-check the last 20 parts from this station."}


def region_box(location, w, h, pad=0.02):
    """Pixel box of one cell of the 3x3 location grid."""
    if location not in LOCATIONS:
        return None
    i = LOCATIONS.index(location); r, c = divmod(i, 3)
    return (max(0, int((c / 3 - pad) * w)), max(0, int((r / 3 - pad) * h)),
            min(w, int(((c + 1) / 3 + pad) * w)), min(h, int(((r + 1) / 3 + pad) * h)))


def _b64(pil, size=384):
    buf = io.BytesIO(); pil.convert("RGB").resize((size, size)).save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


class State:
    def __init__(self):
        t0 = time.perf_counter()
        self.splits = data.make_splits()
        masks = {c: data.load_cache(c, self.splits)[1] for c in CATEGORIES}
        self.items = eval_items(self.splits, masks)
        self.t1, self.t2 = serving.tier1(), serving.tier2()
        self.refs = {c: serving.reference_image(self.splits, c) for c in CATEGORIES}
        self.check = InputCheck(self.splits)
        self.delta = DeltaMap(self.splits)                 # explicit good-vs-part delta (heatmap + score)
        self.act = ActionBuilder(self.t2, self.refs)   # SOP-grounded machine JSON for the line controller
        self.defect_types = {c: data.defect_types(c) for c in CATEGORIES}
        saved = policy.load() or {}
        self.lik = saved.get("likelihoods")
        self.costs = dict(saved.get("costs") or policy.DEFAULT_COSTS)
        self.defect_rate = saved.get("defect_rate", policy.DEFAULT_DEFECT_RATE)
        self.audit_rate = saved.get("audit_rate", 0.05)
        self.t_lo = saved.get("t_lo", 0.05)
        self.privacy = "roi"
        self.recompute_policy()
        self.eval_df = self._load_eval_df()
        self.store = EdgeStore()
        self.sync = CloudSync(self.store, CLOUD_URL, CLOUD_KEY, SITE).start()
        self.images = collections.OrderedDict()          # recent inspections, for chat and ROI
        self.line = LineSimulator(self.t1_fn, self.t2_fn, self.policy_fn, self.t_lo, self.items, parts_per_minute=120,
                                  defect_rate=0.05, audit_rate=self.audit_rate, on_event=self.on_line_event)
        self.bench = json.loads((RESULTS_DIR / "benchmark_summary.json").read_text()) if (RESULTS_DIR / "benchmark_summary.json").exists() else {}
        self._device, self._device_t = {}, 0
        e = self.bench.get("S5_energy", {})
        self.joules = {"t1": (e.get("tier1") or {}).get("joules_per_item", 12.7), "t2": (e.get("tier2") or {}).get("joules_per_item", 51.5)}
        self.extra = {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "energy_j": 0.0}   # chat + NCR calls
        self.rates = {"cloud_model": "GPT-4o-equivalent (as configured in the ZGX Console)", "input_per_m_usd": 2.50, "output_per_m_usd": 10.00,
                      "electricity_per_kwh_usd": 0.15, "uplink_mbps": 20.0, "cloud_rtt_s": 1.0, "parts_per_minute": 60, "hours_per_day": 16}
        self.started = time.time(); self.boot_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------ tiers
    def t1_fn(self, pil, cat):
        return self.t1.safe_ask([pil], cat)

    def t2_fn(self, pil, cat):
        return self.t2.safe_ask([self.refs[cat], pil], cat)

    # ------------------------------------------------------------ policy
    def recompute_policy(self):
        self.table = policy.policy_table(self.lik, self.costs, self.defect_rate) if self.lik else None
        self.actions = self.table.set_index("bucket").to_dict("index") if self.table is not None else {}

    def policy_fn(self, p1, t_lo, verdict2):
        b = policy.bucket(p1, t_lo, verdict2)
        row = self.actions.get(b)
        if row is None:              # no fitted policy yet: accept/reject only when both tiers agree
            if b == "agree_good": return "accept", policy.BUCKETS[b], b
            if b == "agree_defect": return "reject", policy.BUCKETS[b], b
            return "escalate_to_human", policy.BUCKETS[b], b
        why = (f"{policy.BUCKETS[b]}; P(defect) {row['p_defect']:.1%} → expected cost if accepted "
               f"${row['expected_cost_accept_usd']:.2f}, if rejected ${row['expected_cost_reject_usd']:.2f}, "
               f"human review ${row['human_review_usd']:.2f}")
        return row["action"], why, b

    def _load_eval_df(self):
        try:
            t1 = pd.read_csv(RESULTS_DIR / "llm_t1_ft.csv"); t2 = pd.read_csv(RESULTS_DIR / "llm_t2_ref.csv").set_index("path")
            df = t1[["path", "category", "label"]].copy()
            df["score"] = t1.p_defective; df["review_at"] = self.t_lo; df["verdict"] = df.path.map(t2.verdict)
            return df
        except Exception:
            return None

    # ------------------------------------------------------------ record + escalate
    def record(self, *, source, cat, image_ref, decision, tier, bucket, reason, r1, r2, total_s, image_bytes, true_label, pil, hm=None):
        loc = (r2 or {}).get("location") if (r2 or {}).get("verdict") == "defective" else (r1 or {}).get("location")
        use = [r for r in (r1, r2) if r and r.get("prompt_tokens")]
        tok_in = sum(r["prompt_tokens"] or 0 for r in use); tok_out = sum(r.get("completion_tokens") or 0 for r in use)
        energy = (self.joules["t1"] if r1 and r1.get("prompt_tokens") else 0) + (self.joules["t2"] if r2 and r2.get("prompt_tokens") else 0)
        gpu_s = sum(r.get("latency_s") or 0 for r in use)
        iid = self.store.record(tokens_in=tok_in, tokens_out=tok_out, llm_calls=len(use), energy_j=energy, gpu_s=gpu_s,
                                ts=time.time(), source=source, category=cat, image_ref=str(image_ref), decision=decision, tier=tier,
                                bucket=bucket, reason=reason, vision_score=(r1 or {}).get("p_defective"), review_at=self.t_lo,
                                vlm_verdict=(r2 or {}).get("verdict"), vlm_type=(r2 or r1 or {}).get("defect_type"),
                                vlm_location=loc, vision_ms=((r1 or {}).get("latency_s") or 0) * 1000,
                                vlm_s=(r2 or {}).get("latency_s"), total_s=total_s, image_bytes=image_bytes, true_label=true_label)
        self.images[iid] = (pil, cat);
        while len(self.images) > 200: self.images.popitem(last=False)
        eid = None
        if tier == 3 and pil is not None:
            box = DeltaMap.peak_box(hm, *pil.size) if hm is not None else (region_box(loc, *pil.size) or (0, 0, pil.size[0], pil.size[1]))
            crop = pil.crop(box) if self.privacy == "roi" else pil
            buf = io.BytesIO(); crop.convert("RGB").resize((min(256, crop.size[0]), min(256, crop.size[1]))).save(buf, format="JPEG", quality=80)
            payload = {"category": cat, "bucket": bucket, "reason": reason, "vision_score": (r1 or {}).get("p_defective"),
                       "review_at": self.t_lo,
                       "vlm": {"verdict": (r2 or {}).get("verdict"), "defect_type": (r2 or {}).get("defect_type"), "location": loc},
                       "tier1": {k: (r1 or {}).get(k) for k in ("verdict", "defect_type", "location", "p_defective")},
                       "explanation": (r2 or {}).get("explanation"), "edge_proposal": decision,
                       "defect_types": self.defect_types.get(cat, []), "privacy_mode": self.privacy}
            eid = self.store.escalate(iid, payload, buf.getvalue())
        return iid, eid

    def on_line_event(self, ev, pil, r1, r2):
        size = (DATA_ROOT / ev["path"]).stat().st_size
        src = r2 if (r2 or {}).get("verdict") == "defective" else (r1 or {})
        try:   # line events use the SOP template (no extra tier-2 call per part), so throughput is unaffected
            self.act.build(cat=ev["category"], decision=ev["decision"], part_id=f"LINE-{ev['part']:06d}", defect_type=src.get("defect_type"),
                               location=src.get("location"), use_llm=False, evidence={"tier1_p_defect": ev.get("p1"), "evidence_bucket": ev.get("bucket")})
        except Exception:
            pass
        self.record(source="line-audit" if ev.get("audit") else "line", cat=ev["category"], image_ref=ev["path"],
                    decision=ev["decision"], tier=ev["tier"], bucket=ev.get("bucket"), reason=ev.get("reason", ""), r1=r1, r2=r2,
                    total_s=ev.get("total_s"), image_bytes=size, true_label=ev["label"], pil=pil)

    # ------------------------------------------------------------ one part through the cascade
    def cascade(self, pil, cat, source, force_t2=False, image_ref="upload", true_label=None, image_bytes=None):
        audit = random.random() < self.audit_rate
        res = decide_part(pil, cat, t1=self.t1, t2=self.t2, refs=self.refs, check=self.check, policy_fn=self.policy_fn,
                          t_lo=self.t_lo, force_t2=force_t2, audit=audit)
        decision, tier, bucket, reason, r1, r2 = (res[k] for k in ("decision", "tier", "bucket", "reason", "r1", "r2"))
        audit = audit and tier > 1 and (r1 or {}).get("p_defective", 1) < self.t_lo
        pil = pil.convert("RGB"); total = res["total_s"]; chk = res["input_check"]
        dscore, hm, dz = self.delta.score(pil, cat)
        iid, eid = self.record(source=source, cat=cat, image_ref=image_ref, decision=decision, tier=tier, bucket=bucket, reason=reason,
                               r1=r1, r2=r2, total_s=total, image_bytes=image_bytes, true_label=true_label, pil=pil, hm=hm)
        r1 = r1 or {}
        src = r2 if (r2 or {}).get("verdict") == "defective" else r1
        self.act.last_usage = (0, 0)
        machine, machine_src = self.act.build(
            cat=cat, decision=decision, part_id=f"{SITE.split('/')[-1].strip()}-{iid:06d}", defect_type=src.get("defect_type"),
            location=src.get("location"), explanation=(r2 or {}).get("explanation"), pil=pil,
            evidence={"tier1_p_defect": r1.get("p_defective"), "tier2_verdict": (r2 or {}).get("verdict"),
                      "delta_score": round(dscore, 3), "delta_z_vs_good": round(dz, 2), "evidence_bucket": bucket})
        marked = pil.copy(); d = ImageDraw.Draw(marked)
        loc = (r2 or {}).get("location") if (r2 or {}).get("verdict") == "defective" else r1.get("location") if r1.get("verdict") == "defective" else None
        box = region_box(loc, *pil.size)
        if box: d.rectangle(box, outline=(220, 38, 38), width=max(3, pil.size[0] // 120))
        pick = lambda r, ks: {k: r.get(k) for k in ks} if r else None
        if self.act.last_usage[0]:
            self.extra["tokens_in"] += self.act.last_usage[0]; self.extra["tokens_out"] += self.act.last_usage[1]
            self.extra["llm_calls"] += 1; self.extra["energy_j"] += self.joules["t2"]
        heat = (plt_cm(hm / max(hm.max(), 1e-6)) * 255).astype(np.uint8)
        overlay = Image.blend(pil.resize((256, 256)), Image.fromarray(heat), 0.45)
        return {"inspection_id": iid, "escalation_id": eid, "decision": decision, "tier": tier, "bucket": bucket, "reason": reason,
                "delta": {"score": round(dscore, 3), "z_vs_good": round(dz, 2)}, "delta_b64": _b64(overlay),
                "machine_json": machine, "machine_json_source": machine_src,
                "audit": audit, "category": cat, "t_lo": self.t_lo, "total_s": round(total, 3), "input_check": chk,
                "tier1": pick(r1, ("verdict", "defect_type", "location", "p_defective", "latency_s", "valid")),
                "tier2": pick(r2, ("verdict", "defect_type", "location", "explanation", "p_defective", "latency_s", "valid")),
                "recommended_action": SOP[decision].format(cat=cat.replace("_", " ")),
                "image_b64": _b64(pil), "marked_b64": _b64(marked), "reference_b64": _b64(self.refs[cat], 192)}

    def device(self):
        if time.time() - self._device_t > 2:
            try: self._device = telemetry.gpu_sample()
            except Exception: self._device = {}
            self._device_t = time.time()
        return self._device


# ================================================================ HTTP API
app = FastAPI(title="NanoInspect Edge")
S: State = None


@app.on_event("startup")
def _startup():
    global S
    S = State()
    print(f"NanoInspect Edge ready in {S.boot_seconds:.0f}s: site {SITE}; tier 1 {S.t1.url} ({S.t1.healthy()}); "
          f"tier 2 {S.t2.url} ({S.t2.healthy()}); cloud {CLOUD_URL}")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/pitch")
def pitch():
    return FileResponse(WEB / "pitch.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/docs", StaticFiles(directory=DOCS), name="docs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/api/meta")
def meta():
    return {"site": SITE, "categories": CATEGORIES, "defect_types": S.defect_types, "cloud_url": CLOUD_URL, "started": S.started,
            "tier1": {"model": "Qwen2.5-VL-7B-Instruct + NanoInspect LoRA", "served_as": S.t1.model, "url": S.t1.url, "healthy": S.t1.healthy()},
            "tier2": {"model": "Qwen3.8-27B (NVFP4)", "served_as": S.t2.model, "url": S.t2.url, "healthy": S.t2.healthy()},
            "t_lo": S.t_lo}


@app.get("/api/overview")
def overview():
    st = S.store.stats(); b = S.bench.get("S5_energy", {})
    j1 = (b.get("tier1") or {}).get("joules_per_item"); j2 = (b.get("tier2") or {}).get("joules_per_item")
    t = st["by_tier"]
    energy = (st["total"] * j1 + ((t.get("2", 0) + t.get("3", 0)) * j2)) / 3600 if j1 and j2 else None
    return {"stats": st, "sync": S.sync.status(), "device": S.device(), "line": S.line.snapshot(), "energy_wh": energy,
            "privacy": S.privacy, "audit_rate": S.audit_rate}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), category: str = Form(...), force_t2: bool = Form(False), source: str = Form("upload")):
    if category not in CATEGORIES:
        raise HTTPException(400, "unknown product")
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)); pil.load()
    except Exception as exc:
        iid, _ = S.record(source=source, cat=category, image_ref=file.filename, decision="manual_review", tier=3, bucket="unreadable",
                          reason=f"unreadable image: {exc}", r1=None, r2=None, total_s=0, image_bytes=len(raw), true_label=None, pil=None)
        return {"decision": "manual_review", "tier": 3, "bucket": "unreadable", "category": category, "inspection_id": iid,
                "reason": "The image could not be read. Hold the part and check the camera."}
    return S.cascade(pil, category, source, force_t2=force_t2, image_ref=file.filename, image_bytes=len(raw))


class SampleReq(BaseModel):
    category: str
    kind: str = "any"
    force_t2: bool = False


@app.post("/api/inspect_sample")
def inspect_sample(r: SampleReq):
    pool = [i for i in S.items if i["category"] == r.category and (r.kind == "any" or (i["label"] == 1) == (r.kind == "defect"))]
    it = random.choice(pool)
    out = S.cascade(data.load_pil(it["path"]), r.category, "sample", force_t2=r.force_t2, image_ref=it["path"],
                    true_label=it["label"], image_bytes=(DATA_ROOT / it["path"]).stat().st_size)
    out["ground_truth"] = {"label": "defective" if it["label"] else "good", "defect_type": it["true_type"],
                           "location": it["true_location"], "path": it["path"]}
    return out


DEMO_SCENARIOS = [
    {"id": "accept", "title": "Good part", "expect": "accept", "category": "bottle", "path": "bottle/test/good/002.png",
     "story": "Tier 1 is confident the bottle is good, so it is accepted in about 2 seconds. No big model, no person."},
    {"id": "reject", "title": "Clear defect", "expect": "reject", "category": "bottle", "path": "bottle/test/broken_large/000.png",
     "story": "Both LLM tiers see a broken rim. Rejecting is cheaper than asking a person, and the line receives a machine instruction."},
    {"id": "review", "title": "Models disagree", "expect": "manual_review", "category": "bottle", "path": "bottle/test/contamination/009.png",
     "story": "Tier 1 sees contamination, tier 2 does not. The disagreement goes to a person in the cloud; only a crop leaves the plant."},
]


@app.get("/api/demo_scenarios")
def demo_scenarios():
    return DEMO_SCENARIOS


class PathReq(BaseModel):
    category: str
    path: str
    force_t2: bool = False


@app.post("/api/inspect_path")
def inspect_path(r: PathReq):
    p = (DATA_ROOT / r.path).resolve()
    if DATA_ROOT.resolve() not in p.parents or not p.exists() or r.category not in CATEGORIES:
        raise HTTPException(404, "not a dataset image")
    it = next((i for i in S.items if i["path"] == r.path), None)
    out = S.cascade(data.load_pil(r.path), r.category, "demo", force_t2=r.force_t2, image_ref=r.path,
                    true_label=it["label"] if it else None, image_bytes=p.stat().st_size)
    if it:
        out["ground_truth"] = {"label": "defective" if it["label"] else "good", "defect_type": it["true_type"],
                               "location": it["true_location"], "path": it["path"]}
    return out


@app.get("/api/showcase")
def showcase(n: int = 24, seed: int = 7):
    rng = random.Random(seed); cats = list(CATEGORIES); out = []
    for i in range(n):
        c = cats[i % len(cats)]; pool = [x for x in S.items if x["category"] == c]
        it = rng.choice(pool); out.append({"category": c, "path": it["path"], "label": it["label"], "type": it["true_type"]})
    return out


class ChatReq(BaseModel):
    inspection_id: int
    question: str
    history: list = []


@app.post("/api/chat")
def chat(r: ChatReq):
    """Ask tier 2 (Qwen3.8-27B) a follow-up question about an inspected part."""
    if r.inspection_id not in S.images:
        raise HTTPException(404, "inspection not in memory")
    pil, cat = S.images[r.inspection_id]
    msgs = [{"role": "system", "content": "You are NanoInspect's senior quality inspector. Answer briefly and concretely, "
                                         "in at most 4 sentences, about the product image the operator is looking at."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": serving.to_data_url(S.refs[cat])}},
                                         {"type": "image_url", "image_url": {"url": serving.to_data_url(pil)}},
                                         {"type": "text", "text": f"The first image is a known-good {cat.replace('_', ' ')}; "
                                                                  f"the second is the part being inspected. {r.question}"}]}]
    for h in r.history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    if r.history:
        msgs.append({"role": "user", "content": r.question})
    t0 = time.perf_counter()
    resp = S.t2.session.post(f"{S.t2.url}/v1/chat/completions", timeout=120, json={
        "model": S.t2.model, "messages": msgs, "temperature": 0.2, "max_tokens": 220,
        "chat_template_kwargs": {"enable_thinking": False}})
    resp.raise_for_status(); j = resp.json(); u = j.get("usage", {})
    S.extra["tokens_in"] += u.get("prompt_tokens", 0); S.extra["tokens_out"] += u.get("completion_tokens", 0)
    S.extra["llm_calls"] += 1; S.extra["energy_j"] += S.joules["t2"]
    return {"answer": j["choices"][0]["message"]["content"], "latency_s": round(time.perf_counter() - t0, 2),
            "tokens": j.get("usage", {}).get("completion_tokens")}


@app.get("/api/models")
def models():
    bench = S.bench.get("serving", {})
    return {"tier1": {**meta()["tier1"], "metrics": S.t1.metrics(), "benchmark": bench.get("tier1")},
            "tier2": {**meta()["tier2"], "metrics": S.t2.metrics(), "benchmark": bench.get("tier2")},
            "quality": S.bench.get("model_quality"), "device": S.device()}


@app.get("/api/controller")
def controller(n: int = 30):
    c = S.act.controller
    return {"messages": list(c.messages)[:n], "stopped": c.stopped, "sop_id": S.act.sop["sop_id"], "sop_version": S.act.sop["version"]}


@app.post("/api/controller/reset")
def controller_reset():
    S.act.controller.stopped = None; S.act.controller.recent.clear(); return {"ok": True}


@app.get("/api/sop")
def sop():
    return S.act.sop


def _savings(tok_in, tok_out, calls, energy_j, gpu_s, image_bytes, parts, R):
    cloud_in = tok_in / 1e6 * R["input_per_m_usd"]; cloud_out = tok_out / 1e6 * R["output_per_m_usd"]
    kwh = energy_j / 3.6e6; power = kwh * R["electricity_per_kwh_usd"]
    upload_s = image_bytes * 8 / (R["uplink_mbps"] * 1e6)
    return {"parts": parts, "llm_calls_avoided": calls, "tokens_in": tok_in, "tokens_out": tok_out, "tokens_total": tok_in + tok_out,
            "cloud_input_usd": cloud_in, "cloud_output_usd": cloud_out, "cloud_equivalent_usd": cloud_in + cloud_out,
            "energy_kwh": kwh, "electricity_usd": power, "net_saved_usd": cloud_in + cloud_out - power,
            "blended_rate_per_m_usd": (cloud_in + cloud_out) / ((tok_in + tok_out) / 1e6) if tok_in + tok_out else None,
            "local_gpu_s": gpu_s, "image_mb_kept_local": image_bytes / 1e6, "upload_time_avoided_s": upload_s,
            "network_time_avoided_s": upload_s + calls * R["cloud_rtt_s"]}


@app.get("/api/savings")
def savings():
    st = S.store.stats(); u = st["usage"]; R = S.rates; x = S.extra
    live = _savings(u["tokens_in"] + x["tokens_in"], u["tokens_out"] + x["tokens_out"], u["llm_calls"] + x["llm_calls"],
                    u["energy_j"] + x["energy_j"], u["gpu_s"], st["bytes_inspected"] - st["bytes_sent_to_cloud"], st["total"], R)
    # Projection from the held-out evaluation: measured tokens per call and the cascade's tier-2 call rate
    proj = None
    try:
        t1 = pd.read_csv(RESULTS_DIR / "llm_t1_ft.csv"); t2 = pd.read_csv(RESULTS_DIR / "llm_t2_ref.csv")
        t2_rate = (policy.load() or {}).get("held_out_result", {}).get("NanoInspect cascade", {}).get("vlm_calls_per_1000", 148) / 1000
        n = R["parts_per_minute"] * 60 * R["hours_per_day"]
        tin = n * (t1.prompt_tokens.mean() + t2_rate * t2.prompt_tokens.mean()); tout = n * (t1.completion_tokens.mean() + t2_rate * t2.completion_tokens.mean())
        idle_w = S.bench.get("S5_energy", {}).get("idle_gpu_power_w", 12.4)
        e = n * (S.joules["t1"] + t2_rate * S.joules["t2"]) + idle_w * R["hours_per_day"] * 3600   # inference + idle GPU for the shift
        mb = np.mean([(DATA_ROOT / p).stat().st_size for p in t1.path[:200]])
        proj = {**_savings(tin, tout, n * (1 + t2_rate), e, None, n * mb, n, R),
                "per_call_tokens": {"tier1_in": round(t1.prompt_tokens.mean()), "tier1_out": round(t1.completion_tokens.mean()),
                                    "tier2_in": round(t2.prompt_tokens.mean()), "tier2_out": round(t2.completion_tokens.mean())},
                "tier2_call_rate": t2_rate}
    except Exception as exc:
        proj = {"error": str(exc)}
    return {"rates": R, "live": live, "per_day_projection": proj}


class RatesReq(BaseModel):
    input_per_m_usd: float
    output_per_m_usd: float
    electricity_per_kwh_usd: float
    uplink_mbps: float = 20.0
    cloud_rtt_s: float = 1.0
    parts_per_minute: int = 60
    hours_per_day: float = 16


@app.post("/api/savings/rates")
def set_rates(r: RatesReq):
    S.rates.update(r.model_dump()); return savings()


@app.get("/api/thumb")
def thumb(path: str, size: int = 160):
    p = (DATA_ROOT / path).resolve()
    if DATA_ROOT.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    buf = io.BytesIO(); Image.open(p).convert("RGB").resize((size, size)).save(buf, format="JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/recent")
def recent(n: int = 30):
    return S.store.recent(n)


class LineReq(BaseModel):
    parts_per_minute: int = 120
    defect_rate: float = 0.05


@app.post("/api/line/start")
def line_start(r: LineReq):
    S.line.parts_per_minute, S.line.defect_rate, S.line.audit_rate = r.parts_per_minute, r.defect_rate, S.audit_rate
    S.line.start(); return S.line.snapshot()


@app.post("/api/line/stop")
def line_stop():
    threading.Thread(target=S.line.stop, daemon=True).start(); return {"stopping": True}


@app.get("/api/line")
def line_state():
    return {"snapshot": S.line.snapshot(), "timeline": S.line.timeline[-300:], "events": list(S.line.events)[:24],
            "last_flagged": S.line.last_flagged}


@app.get("/api/escalations")
def escalations(n: int = 60):
    return {"items": S.store.escalations(n), "sync": S.sync.status()}


@app.get("/api/escalations/{eid}/roi")
def esc_roi(eid: str):
    b = S.store.roi(eid)
    if b is None: raise HTTPException(404)
    return Response(b, media_type="image/jpeg")


class NetReq(BaseModel):
    online: bool


@app.post("/api/network")
def network(r: NetReq):
    S.sync.online = r.online; return S.sync.status()


@app.get("/api/policy")
def get_policy():
    saved = policy.load() or {}
    return {"costs": S.costs, "defect_rate": S.defect_rate, "audit_rate": S.audit_rate, "privacy": S.privacy, "t_lo": S.t_lo,
            "table": S.table.round(5).to_dict("records") if S.table is not None else [],
            "held_out": saved.get("held_out_result"), "fitted": S.lik is not None}


class PolicyReq(BaseModel):
    escape_usd: float
    false_reject_usd: float
    human_review_usd: float
    defect_rate: float
    audit_rate: float = 0.05
    privacy: str = "roi"


@app.post("/api/policy")
def set_policy(r: PolicyReq):
    S.costs = {"escape_usd": r.escape_usd, "false_reject_usd": r.false_reject_usd, "human_review_usd": r.human_review_usd}
    S.defect_rate = min(max(r.defect_rate, 0.0005), 0.5); S.audit_rate = min(max(r.audit_rate, 0.0), 0.5)
    S.privacy = r.privacy if r.privacy in ("roi", "full") else "roi"
    S.line.audit_rate = S.audit_rate
    S.recompute_policy()
    return get_policy() | {"simulation": simulate_now()}


@app.get("/api/policy/simulate")
def simulate_now():
    if S.eval_df is None or S.table is None:
        return None
    res = {"NanoInspect cascade": policy.simulate(S.eval_df, S.table, S.defect_rate, S.costs, S.audit_rate),
           **policy.baselines(S.eval_df, S.defect_rate, S.costs)}
    return {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}


@app.get("/api/benchmarks")
def benchmarks():
    return {"summary": S.bench, "figures": sorted(p.name for p in RESULTS_DIR.glob("*.png"))}


@app.post("/api/reset")
def reset():
    S.line.stop(); S.store.reset(); S.line.reset(); return {"ok": True}


def main():
    uvicorn.run(app, host=os.environ.get("NANOINSPECT_HOST", "127.0.0.1"), port=int(os.environ.get("NANOINSPECT_PORT", 8080)))


if __name__ == "__main__":
    main()
