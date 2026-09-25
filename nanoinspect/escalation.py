"""Edge-side record keeping and store-and-forward escalation to the cloud review service.

Every inspection is written to a local SQLite database on the Nano. Parts the policy escalates get
an outbox entry holding only what a reviewer needs: a small crop around the region the vision model
looked at, plus the edge models' verdicts. A sync thread sends queued items whenever the cloud is
reachable and pulls human labels back. If the network is down, the line keeps running and the
outbox simply grows until connectivity returns.
"""
import base64
import io
import json
import sqlite3
import threading
import time
import uuid

import numpy as np
from PIL import Image

from .config import ARTIFACTS

DB_PATH = ARTIFACTS / "edge.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, source TEXT, category TEXT, image_ref TEXT,
  decision TEXT, tier INTEGER, bucket TEXT, reason TEXT, vision_score REAL, review_at REAL,
  vlm_verdict TEXT, vlm_type TEXT, vlm_location TEXT, vision_ms REAL, vlm_s REAL, total_s REAL,
  image_bytes INTEGER, escalated INTEGER DEFAULT 0, true_label INTEGER, human_verdict TEXT);
CREATE TABLE IF NOT EXISTS escalations (
  id TEXT PRIMARY KEY, inspection_id INTEGER, created REAL, status TEXT, attempts INTEGER DEFAULT 0,
  last_error TEXT, sent_at REAL, payload TEXT, roi BLOB, bytes INTEGER,
  human_verdict TEXT, human_defect_type TEXT, reviewer TEXT, reviewed_at REAL);
CREATE INDEX IF NOT EXISTS idx_insp_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS idx_esc_status ON escalations(status);
"""


def roi_crop(pil, cam, out_size=192, frac=0.35):
    """Crop the square around the Grad-CAM peak (frac of the image side), as a small JPEG."""
    w, h = pil.size
    if cam is not None:
        y, x = np.unravel_index(np.argmax(cam), cam.shape)
        cx, cy = x / cam.shape[1] * w, y / cam.shape[0] * h
    else:
        cx, cy = w / 2, h / 2
    side = int(min(w, h) * frac)
    left = int(np.clip(cx - side / 2, 0, w - side)); top = int(np.clip(cy - side / 2, 0, h - side))
    crop = pil.crop((left, top, left + side, top + side)).resize((out_size, out_size), Image.BICUBIC)
    buf = io.BytesIO(); crop.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue(), {"left": left / w, "top": top / h, "size": side / w}


class EdgeStore:
    def __init__(self, path=DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for col, typ in (("tokens_in", "INTEGER"), ("tokens_out", "INTEGER"), ("llm_calls", "INTEGER"), ("energy_j", "REAL"), ("gpu_s", "REAL")):
            try: self.db.execute(f"ALTER TABLE inspections ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError: pass          # column already exists
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.db.executescript("DELETE FROM inspections; DELETE FROM escalations;"); self.db.commit()

    def record(self, **f):
        cols = ",".join(f); q = ",".join("?" * len(f))
        with self.lock:
            cur = self.db.execute(f"INSERT INTO inspections ({cols}) VALUES ({q})", list(f.values())); self.db.commit()
            return cur.lastrowid

    def escalate(self, inspection_id, payload, roi_jpeg):
        eid = uuid.uuid4().hex[:12]
        body = json.dumps(payload)
        with self.lock:
            self.db.execute("INSERT INTO escalations (id, inspection_id, created, status, payload, roi, bytes) VALUES (?,?,?,?,?,?,?)",
                            (eid, inspection_id, time.time(), "queued", body, roi_jpeg, len(body) + len(roi_jpeg)))
            self.db.execute("UPDATE inspections SET escalated=1 WHERE id=?", (inspection_id,)); self.db.commit()
        return eid

    def queued(self, limit=20):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM escalations WHERE status='queued' ORDER BY created LIMIT ?", (limit,))]

    def mark(self, eid, **f):
        sets = ",".join(f"{k}=?" for k in f)
        with self.lock:
            self.db.execute(f"UPDATE escalations SET {sets} WHERE id=?", [*f.values(), eid]); self.db.commit()

    def apply_label(self, eid, verdict, defect_type, reviewer, reviewed_at):
        with self.lock:
            self.db.execute("UPDATE escalations SET status='reviewed', human_verdict=?, human_defect_type=?, reviewer=?, reviewed_at=? WHERE id=?",
                            (verdict, defect_type, reviewer, reviewed_at, eid))
            self.db.execute("UPDATE inspections SET human_verdict=? WHERE id=(SELECT inspection_id FROM escalations WHERE id=?)", (verdict, eid))
            self.db.commit()

    def recent(self, n=30):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (n,))]

    def escalations(self, n=50):
        with self.lock:
            rows = self.db.execute("SELECT e.id, e.created, e.status, e.attempts, e.last_error, e.sent_at, e.bytes, e.human_verdict, "
                                   "e.human_defect_type, e.reviewer, e.reviewed_at, i.category, i.bucket, i.vision_score, i.vlm_verdict, "
                                   "i.vlm_type, i.source FROM escalations e JOIN inspections i ON i.id=e.inspection_id "
                                   "ORDER BY e.created DESC LIMIT ?", (n,))
            return [dict(r) for r in rows]

    def roi(self, eid):
        with self.lock:
            r = self.db.execute("SELECT roi FROM escalations WHERE id=?", (eid,)).fetchone()
        return r["roi"] if r else None

    def stats(self):
        with self.lock:
            q = lambda s, *a: self.db.execute(s, a).fetchone()[0]
            total = q("SELECT COUNT(*) FROM inspections")
            by_dec = {r[0]: r[1] for r in self.db.execute("SELECT decision, COUNT(*) FROM inspections GROUP BY decision")}
            by_tier = {r[0]: r[1] for r in self.db.execute("SELECT tier, COUNT(*) FROM inspections GROUP BY tier")}
            by_status = {r[0]: r[1] for r in self.db.execute("SELECT status, COUNT(*) FROM escalations GROUP BY status")}
            sent_bytes = q("SELECT COALESCE(SUM(bytes),0) FROM escalations WHERE status IN ('sent','reviewed')")
            all_bytes = q("SELECT COALESCE(SUM(image_bytes),0) FROM inspections")
            lat = [r[0] for r in self.db.execute("SELECT vision_ms FROM inspections WHERE vision_ms IS NOT NULL ORDER BY id DESC LIMIT 2000")]
            vlat = [r[0] for r in self.db.execute("SELECT vlm_s FROM inspections WHERE vlm_s IS NOT NULL ORDER BY id DESC LIMIT 500")]
            labelled = q("SELECT COUNT(*) FROM escalations WHERE status='reviewed'")
            agree = q("SELECT COUNT(*) FROM escalations e JOIN inspections i ON i.id=e.inspection_id WHERE e.status='reviewed' AND "
                      "((e.human_verdict='defective' AND i.vlm_verdict='defective') OR (e.human_verdict='good' AND i.vlm_verdict='good'))")
            esc_truth = q("SELECT COUNT(*) FROM inspections WHERE true_label=1 AND decision='accept'")
            fr_truth = q("SELECT COUNT(*) FROM inspections WHERE true_label=0 AND decision='reject'")
            row = self.db.execute("SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), COALESCE(SUM(llm_calls),0), "
                                  "COALESCE(SUM(energy_j),0), COALESCE(SUM(gpu_s),0) FROM inspections").fetchone()
            usage = dict(zip(("tokens_in", "tokens_out", "llm_calls", "energy_j", "gpu_s"), tuple(row)))
            by_cat = [dict(r) for r in self.db.execute(
                "SELECT category, COUNT(*) AS parts, SUM(decision='reject') AS rejected, SUM(decision='manual_review') AS review "
                "FROM inspections GROUP BY category ORDER BY parts DESC")]
        pct = lambda xs, p: float(np.percentile(xs, p)) if xs else None
        return {"total": total, "by_decision": by_dec, "by_tier": {str(k): v for k, v in by_tier.items()},
                "escalations": by_status, "bytes_sent_to_cloud": sent_bytes, "bytes_inspected": all_bytes,
                "vision_p50_ms": pct(lat, 50), "vision_p95_ms": pct(lat, 95), "vlm_p50_s": pct(vlat, 50), "vlm_p95_s": pct(vlat, 95),
                "human_labels": labelled, "human_vlm_agreement": (agree / labelled) if labelled else None,
                "escapes_ground_truth": esc_truth, "false_rejects_ground_truth": fr_truth, "by_category": by_cat, "usage": usage}


class CloudSync:
    """Background store-and-forward. `online=False` simulates a network cut (for the demo)."""

    def __init__(self, store, cloud_url, api_key, site_id, interval_s=2.0):
        self.store, self.cloud_url, self.api_key, self.site_id = store, cloud_url.rstrip("/"), api_key, site_id
        self.interval_s = interval_s; self.online = True; self.last_ok = None; self.last_error = None
        self._since = 0.0; self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start(); return self

    def _headers(self):
        return {"X-NanoInspect-Key": self.api_key, "X-NanoInspect-Site": self.site_id}

    def _loop(self):
        import requests
        while not self._stop.is_set():
            if self.online:
                try:
                    for e in self.store.queued():
                        body = {"id": e["id"], "site": self.site_id, "created": e["created"], **json.loads(e["payload"]),
                                "roi_jpeg_b64": base64.b64encode(e["roi"]).decode()}
                        r = requests.post(f"{self.cloud_url}/api/escalations", json=body, headers=self._headers(), timeout=5)
                        r.raise_for_status()
                        self.store.mark(e["id"], status="sent", sent_at=time.time(), attempts=e["attempts"] + 1)
                    r = requests.get(f"{self.cloud_url}/api/labels", params={"site": self.site_id, "since": self._since},
                                     headers=self._headers(), timeout=5)
                    r.raise_for_status()
                    for lab in r.json():
                        self.store.apply_label(lab["id"], lab["verdict"], lab.get("defect_type"), lab.get("reviewer"), lab["reviewed_at"])
                        self._since = max(self._since, lab["reviewed_at"])
                    self.last_ok, self.last_error = time.time(), None
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                    for e in self.store.queued(limit=1):
                        self.store.mark(e["id"], attempts=e["attempts"] + 1, last_error=self.last_error)
            self._stop.wait(self.interval_s)

    def status(self):
        return {"online": self.online, "cloud_url": self.cloud_url, "last_ok": self.last_ok, "last_error": self.last_error,
                "reachable": self.online and self.last_error is None and self.last_ok is not None}
