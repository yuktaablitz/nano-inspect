"""Stress tests for the served LLM cascade on the ZGX Nano."""
import io
import os
import random
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

from . import serving
from .config import DATA_ROOT
from .data import load_pil
from .telemetry import GpuTelemetry, measure_energy


def soak(t1, t2, refs, items, minutes, t1_clients=16, t2_clients=6, window_s=30, log=print):
    """Both served models under continuous concurrent load for `minutes`, with GPU telemetry."""
    stop = threading.Event(); lock = threading.Lock()
    counts = {"t1": 0, "t2": 0, "t1_err": 0, "t2_err": 0}
    lat = {"t1": [], "t2": []}
    pils = {it["path"]: load_pil(it["path"]) for it in items}

    def worker(tier):
        rng = random.Random(threading.get_ident())
        while not stop.is_set():
            it = rng.choice(items); im = pils[it["path"]]
            r = (t1.safe_ask([im], it["category"]) if tier == "t1" else t2.safe_ask([refs[it["category"]], im], it["category"]))
            with lock:
                if r.get("latency_s") is None: counts[tier + "_err"] += 1
                else: counts[tier] += 1; lat[tier].append((time.perf_counter(), r["latency_s"]))

    windows = []
    with GpuTelemetry(2.0) as tel:
        threads = [threading.Thread(target=worker, args=("t1",), daemon=True) for _ in range(t1_clients)] + \
                  [threading.Thread(target=worker, args=("t2",), daemon=True) for _ in range(t2_clients)]
        for t in threads: t.start()
        t_end = time.perf_counter() + minutes * 60
        while time.perf_counter() < t_end:
            w0 = time.perf_counter(); c0 = dict(counts)
            time.sleep(min(window_s, max(0.0, t_end - w0)))
            dt = time.perf_counter() - w0
            def p95(k):
                xs = [l for t, l in lat[k] if t >= w0]; return float(np.percentile(xs, 95)) if xs else np.nan
            windows.append({"t_min": (time.perf_counter() - tel.t0) / 60, "t1_img_per_s": (counts["t1"] - c0["t1"]) / dt,
                            "t2_img_per_s": (counts["t2"] - c0["t2"]) / dt, "t1_p95_s": p95("t1"), "t2_p95_s": p95("t2")})
            last = tel.rows[-1] if tel.rows else {}
            log(f"\r{windows[-1]['t_min']:.1f}/{minutes} min | tier 1 {windows[-1]['t1_img_per_s']:.2f} img/s | "
                f"tier 2 {windows[-1]['t2_img_per_s']:.2f} img/s | {last.get('temp_c', '?')} C | {last.get('power_w', '?')} W", end="")
        stop.set()
        for t in threads: t.join(timeout=180)
    log("")
    w, g = pd.DataFrame(windows), tel.df
    def change(col):
        a, b = w[col].iloc[:2].mean(), w[col].iloc[-2:].mean()
        return round((b - a) / a * 100, 2) if a else np.nan
    return {"minutes": minutes, "tier1_inferences": counts["t1"], "tier2_inferences": counts["t2"],
            "errors": counts["t1_err"] + counts["t2_err"],
            "tier1_mean_img_per_s": round(float(w.t1_img_per_s.mean()), 2), "tier2_mean_img_per_s": round(float(w.t2_img_per_s.mean()), 2),
            "tier1_throughput_change_pct": change("t1_img_per_s"), "tier2_throughput_change_pct": change("t2_img_per_s"),
            "max_temp_c": float(g.temp_c.max()), "mean_power_w": round(float(g.power_w.mean()), 1), "max_power_w": float(g.power_w.max()),
            "min_sm_clock_mhz": float(g.sm_clock_mhz.min()), "max_sm_clock_mhz": float(g.sm_clock_mhz.max()),
            "throttle_flag_seen": bool(g.throttle_active.any()), "peak_system_ram_gb": round(float(g.system_ram_used_gb.max()), 1)}, w, g


def energy(t1, t2, refs, items, idle_w, n1=128, n2=48, c1=16, c2=6):
    """GPU energy per inspection for each tier at its serving concurrency, plus single-request latency mode."""
    its1, its2 = items[:n1], items[:n2]
    p1 = [load_pil(i["path"]) for i in its1]; p2 = [load_pil(i["path"]) for i in its2]
    run1 = lambda: serving.run_many(lambda im, c: t1.ask([im], c), [(p, i["category"]) for p, i in zip(p1, its1)], workers=c1)
    run2 = lambda: serving.run_many(lambda im, c: t2.ask([refs[c], im], c), [(p, i["category"]) for p, i in zip(p2, its2)], workers=c2)
    return {"tier1": measure_energy(run1, n1, idle_w), "tier2": measure_energy(run2, n2, idle_w)}


def _jpeg(im, q):
    b = io.BytesIO(); im.save(b, format="JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


PERTURBATIONS = {
    "none": lambda im: im,
    "darker (x0.6)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "brighter (x1.4)": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
    "blur (r=2)": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "JPEG q=30": lambda im: _jpeg(im, 30),
    "rotate 5 deg": lambda im: im.rotate(5),
}


def robustness(t1, items, workers=24, log=print):
    """Tier 1 under camera changes, on a stratified subset of held-out images."""
    rows = []
    pils = [load_pil(it["path"]) for it in items]; y = np.array([it["label"] for it in items])
    for name, fn in PERTURBATIONS.items():
        res = serving.run_many(lambda im, c: t1.safe_ask([fn(im)], c), [(p, it["category"]) for p, it in zip(pils, items)], workers=workers)
        s = np.array([r["p_defective"] for r in res]); pred = np.array([r["verdict"] == "defective" for r in res])
        rows.append({"perturbation": name, "roc_auc": roc_auc_score(y, s), "recall": pred[y == 1].mean(),
                     "false_positive_rate": pred[y == 0].mean(), "valid_json_rate": np.mean([bool(r.get("valid")) for r in res])})
        log(f"{name}: AUC {rows[-1]['roc_auc']:.3f}")
    return pd.DataFrame(rows)


LOCAL = ("127.", "::1", "localhost", "0.0.0.0")


def offline_test(run_case, cases):
    """Block every non-local socket and DNS lookup in this process, force Hugging Face offline mode,
    then run full inspections. The served models are on 127.0.0.1, so only local connections remain."""
    attempts = []
    real_connect, real_gai = socket.socket.connect, socket.getaddrinfo
    def is_local(h): return isinstance(h, str) and h.startswith(LOCAL)
    def guard_connect(self, addr):
        if self.family == socket.AF_UNIX or is_local(addr[0]): return real_connect(self, addr)
        attempts.append(("connect", str(addr))); raise OSError(f"[offline test] blocked connect to {addr}")
    def guard_gai(host, *a, **k):
        if host is None or is_local(host): return real_gai(host, *a, **k)
        attempts.append(("dns", str(host))); raise socket.gaierror(f"[offline test] blocked DNS lookup for {host}")
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    rows = []
    try:
        socket.socket.connect, socket.getaddrinfo = guard_connect, guard_gai
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        for name, (path, cat) in cases.items():
            r = run_case(load_pil(path), cat)
            rows.append({"case": name, "decision": r["decision"], "tier": r["tier"], "bucket": r["bucket"], "seconds": round(r["total_s"], 2),
                         "tier1_answered": bool((r["r1"] or {}).get("valid")), "tier2_answered": bool((r["r2"] or {}).get("valid")) if r["r2"] else None})
    finally:
        socket.socket.connect, socket.getaddrinfo = real_connect, real_gai
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return pd.DataFrame(rows), attempts


def failure_recovery(run_case, work_dir, good_path, defect_path, category, t1, t2):
    d = Path(work_dir); d.mkdir(parents=True, exist_ok=True)
    (d / "random_bytes.png").write_bytes(os.urandom(4096))
    (d / "truncated.png").write_bytes((DATA_ROOT / good_path).read_bytes()[:20000])
    Image.new("RGB", (900, 900), (0, 0, 0)).save(d / "all_black.png")
    Image.new("RGB", (900, 900), (255, 255, 255)).save(d / "all_white.png")
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (900, 900, 3), dtype=np.uint8)).save(d / "pure_noise.png")
    rows = []
    for name, p in [("random bytes", d / "random_bytes.png"), ("truncated PNG", d / "truncated.png"),
                    ("all-black frame (lens cap)", d / "all_black.png"), ("all-white frame (overexposed)", d / "all_white.png"),
                    ("pure noise", d / "pure_noise.png")]:
        try:
            pil = Image.open(p)
        except Exception as exc:
            rows.append({"case": name, "decision": "manual_review", "reason": f"unreadable ({type(exc).__name__})"[:90]}); continue
        r = run_case(pil, category)
        rows.append({"case": name, "decision": r["decision"], "reason": r["reason"][:90]})
    # model servers failing: point a tier at a dead port
    for name, tier in [("tier 2 server down", t2), ("tier 1 server down", t1)]:
        real = tier.url; tier.url = "http://127.0.0.1:9"
        try:
            for label, p in [("good part", good_path), ("defective part", defect_path)]:
                r = run_case(load_pil(p), category)
                rows.append({"case": f"{name} ({label})", "decision": r["decision"], "reason": r["reason"][:90]})
        finally:
            tier.url = real
    return pd.DataFrame(rows)
