"""Production-line simulator for the two-LLM cascade.

Parts arrive at a fixed rate. Tier-1 workers (fine-tuned Qwen2.5-VL-7B) check every part concurrently.
Parts tier 1 is confident about are accepted; the rest, plus a random audit sample of accepted parts,
go to tier-2 workers (Qwen3.8-27B with a known-good reference image). The escalation policy then
accepts, rejects, or sends the part to a human in the cloud. Ground truth only counts escapes.
"""
import collections
import queue
import random
import threading
import time

import numpy as np
import pandas as pd

from .data import load_pil

DECISION_OF = {"accept": "accept", "reject": "reject", "escalate_to_human": "manual_review"}


class LineSimulator:
    def __init__(self, tier1_fn, tier2_fn, policy_fn, t_lo, items, parts_per_minute=120, defect_rate=0.05,
                 audit_rate=0.05, t1_workers=16, t2_workers=6, seed=0, on_event=None):
        """tier1_fn(pil, cat) -> answer dict with p_defective; tier2_fn(pil, cat) -> answer dict with verdict;
        policy_fn(p1, t_lo, verdict2) -> (action, reason, bucket); on_event(event, pil, r1, r2)."""
        self.tier1_fn, self.tier2_fn, self.policy_fn, self.t_lo = tier1_fn, tier2_fn, policy_fn, t_lo
        self.good = [it for it in items if it["label"] == 0]
        self.bad = [it for it in items if it["label"] == 1]
        self.parts_per_minute, self.defect_rate, self.audit_rate = parts_per_minute, defect_rate, audit_rate
        self.t1_workers, self.t2_workers = t1_workers, t2_workers
        self.on_event = on_event
        self.rng = random.Random(seed)
        self._stop = threading.Event(); self._threads = []
        self.reset()

    def reset(self):
        self.q1, self.q2 = queue.Queue(), queue.Queue()
        self.events = collections.deque(maxlen=300)
        self.t1_s, self.t2_s, self.total_s = [], [], []
        self.counts = collections.Counter()
        self.timeline = []
        self.last_flagged = None
        self.t0 = None
        self.lock = threading.Lock()

    def _emit(self, ev, pil=None, r1=None, r2=None):
        with self.lock:
            self.counts[ev["decision"]] += 1
            self.counts[f"tier{ev['tier']}"] += 1
            if ev["label"] == 1 and ev["decision"] == "accept": self.counts["escapes"] += 1
            if ev["label"] == 0 and ev["decision"] == "reject": self.counts["false_rejects"] += 1
            self.events.appendleft({k: ev.get(k) for k in ("part", "category", "label", "p1", "decision", "tier", "bucket",
                                                             "t2", "total_s", "path")})
        if self.on_event:
            try: self.on_event(ev, pil, r1, r2)
            except Exception as exc: self.counts["callback_errors"] += 1; self.last_error = repr(exc)

    def _producer(self):
        next_t = time.perf_counter(); part_id = 0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.01, next_t - now)); continue
            next_t += 60.0 / self.parts_per_minute; part_id += 1
            it = self.rng.choice(self.bad if self.rng.random() < self.defect_rate else self.good)
            self.counts["parts"] += 1
            self.q1.put({"part": part_id, "category": it["category"], "label": it["label"], "path": it["path"],
                         "arrived": time.perf_counter()})

    def _tier1_worker(self):
        while not self._stop.is_set():
            try: ev = self.q1.get(timeout=0.2)
            except queue.Empty: continue
            pil = load_pil(ev["path"])
            r1 = self.tier1_fn(pil, ev["category"])
            if r1.get("latency_s"): self.t1_s.append(r1["latency_s"])
            ev["p1"] = round(float(r1["p_defective"]), 4); ev["t1"] = r1
            if ev["p1"] < self.t_lo:
                done = {**ev, "decision": "accept", "tier": 1, "bucket": "fast_accept",
                        "reason": "tier 1 is confident the part is good", "total_s": round(time.perf_counter() - ev["arrived"], 2)}
                self.total_s.append(done["total_s"])
                self._emit(done, pil, r1, None)
                if self.rng.random() < self.audit_rate:
                    self.counts["audit_queued"] += 1; self.q2.put(({**ev, "audit": True}, pil))
            else:
                self.counts["sent_to_t2"] += 1
                self.q2.put(({**ev, "audit": False}, pil))

    def _tier2_worker(self):
        while not self._stop.is_set():
            try: ev, pil = self.q2.get(timeout=0.2)
            except queue.Empty: continue
            r2 = self.tier2_fn(pil, ev["category"])
            if r2.get("latency_s"): self.t2_s.append(r2["latency_s"])
            verdict = r2["verdict"] if r2.get("valid") else None
            action, why, bkt = self.policy_fn(ev["p1"], self.t_lo, verdict)
            t2txt = f"{r2.get('verdict')}: {r2.get('defect_type')} @ {r2.get('location')}"
            if ev["audit"]:
                self.counts["audited"] += 1
                if action != "accept":
                    self.counts["audit_alerts"] += 1
                    self._emit({**ev, "decision": DECISION_OF[action], "tier": 3 if action == "escalate_to_human" else 2,
                                "bucket": bkt, "reason": "audit: " + why, "t2": t2txt, "audit": True}, pil, ev["t1"], r2)
                continue
            total = round(time.perf_counter() - ev["arrived"], 2); self.total_s.append(total)
            full = {**ev, "decision": DECISION_OF[action], "tier": 3 if action == "escalate_to_human" else 2,
                    "bucket": bkt, "reason": why, "t2": t2txt, "total_s": total}
            self._emit(full, pil, ev["t1"], r2)
            self.last_flagged = {"path": ev["path"], "category": ev["category"], "decision": full["decision"], "reason": why,
                                 "p1": ev["p1"], "t1": {k: ev["t1"].get(k) for k in ("verdict", "defect_type", "location")},
                                 "t2": {k: r2.get(k) for k in ("verdict", "defect_type", "location", "explanation")}}

    def _sampler(self):
        while not self._stop.is_set():
            self.timeline.append({"t_s": round(time.perf_counter() - self.t0, 1), "queue_t1": self.q1.qsize(),
                                  "queue_t2": self.q2.qsize(), "parts": self.counts["parts"],
                                  "decided": sum(self.counts[d] for d in ("accept", "reject", "manual_review"))})
            self._stop.wait(1.0)

    def start(self):
        if self.running: return
        self.reset(); self._stop.clear(); self.t0 = time.perf_counter()
        fns = [self._producer, self._sampler] + [self._tier1_worker] * self.t1_workers + [self._tier2_worker] * self.t2_workers
        self._threads = [threading.Thread(target=f, daemon=True) for f in fns]
        for t in self._threads: t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads: t.join(timeout=60)
        self._threads = []

    @property
    def running(self):
        return any(t.is_alive() for t in self._threads)

    def run_for(self, seconds):
        self.start(); time.sleep(seconds); self.stop()
        return self.snapshot(), pd.DataFrame(self.timeline)

    def snapshot(self):
        el = (time.perf_counter() - self.t0) if self.t0 else 0.0
        c = self.counts
        def pct(xs, q): return round(float(np.percentile(xs, q)), 2) if xs else None
        decided = c["accept"] + c["reject"] + c["manual_review"]
        return {"running": self.running, "elapsed_s": round(el, 1), "target_parts_per_min": self.parts_per_minute,
                "arrived_per_min": round(c["parts"] / el * 60, 1) if el else 0.0,
                "decided_per_min": round(decided / el * 60, 1) if el else 0.0,
                "parts": c["parts"], "decided": decided, "accepted": c["accept"], "rejected": c["reject"],
                "manual_review": c["manual_review"], "tier1": c["tier1"], "tier2": c["tier2"], "tier3": c["tier3"],
                "waiting_tier1": self.q1.qsize(), "waiting_tier2": self.q2.qsize(),
                "tier1_p50_s": pct(self.t1_s, 50), "tier1_p95_s": pct(self.t1_s, 95),
                "tier2_p50_s": pct(self.t2_s, 50), "tier2_p95_s": pct(self.t2_s, 95),
                "decision_p95_s": pct(self.total_s, 95),
                "share_sent_to_t2": round(c["sent_to_t2"] / max(1, c["parts"] - self.q1.qsize()), 3) if c["parts"] else 0.0,
                "escapes": c["escapes"], "false_rejects": c["false_rejects"], "audited": c["audited"], "audit_alerts": c["audit_alerts"]}
