"""NanoInspect Cloud: the escalation tier. It runs NO AI inference. It receives the few cases the
edge could not decide cheaply, gives a human reviewer a console, keeps the fleet-wide defect
library, and returns labels to each edge site for on-device retraining.

Run:  uvicorn cloud.server:app --host 0.0.0.0 --port 9000
Env:  NANOINSPECT_CLOUD_KEY (shared secret the edge sends), NANOINSPECT_CLOUD_DB (sqlite path)
"""
import base64
import os
import sqlite3
import statistics
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

KEY = os.environ.get("NANOINSPECT_CLOUD_KEY", "nanoinspect-demo-key")
DB = Path(os.environ.get("NANOINSPECT_CLOUD_DB", Path(__file__).parent / "cloud.db"))
WEB = Path(__file__).parent / "web"

db = sqlite3.connect(str(DB), check_same_thread=False); db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, site TEXT, created REAL, received REAL, category TEXT, bucket TEXT, reason TEXT,
  vision_score REAL, review_at REAL, vlm_verdict TEXT, vlm_type TEXT, vlm_location TEXT, edge_proposal TEXT,
  defect_types TEXT, roi BLOB, roi_bytes INTEGER, status TEXT DEFAULT 'pending',
  verdict TEXT, defect_type TEXT, reviewer TEXT, reviewed_at REAL, note TEXT);
""")
lock = threading.Lock()
app = FastAPI(title="NanoInspect Cloud (escalation tier)")


def _auth(key):
    if key != KEY:
        raise HTTPException(401, "bad key")


@app.post("/api/escalations")
async def receive(req: Request, x_nanoinspect_key: str = Header(None)):
    _auth(x_nanoinspect_key)
    b = await req.json()
    roi = base64.b64decode(b.get("roi_jpeg_b64", ""))
    vlm = b.get("vlm") or {}
    with lock:
        db.execute("INSERT OR IGNORE INTO items (id, site, created, received, category, bucket, reason, vision_score, review_at, "
                   "vlm_verdict, vlm_type, vlm_location, edge_proposal, defect_types, roi, roi_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (b["id"], b.get("site"), b.get("created"), time.time(), b.get("category"), b.get("bucket"), b.get("reason"),
                    b.get("vision_score"), b.get("review_at"), vlm.get("verdict"), vlm.get("defect_type"), vlm.get("location"),
                    b.get("edge_proposal"), ",".join(b.get("defect_types", [])), roi, len(roi)))
        db.commit()
    return {"ok": True}


@app.get("/api/queue")
def queue(status: str = "pending", limit: int = 100):
    with lock:
        rows = db.execute("SELECT id, site, created, received, category, bucket, reason, vision_score, review_at, vlm_verdict, vlm_type, "
                          "vlm_location, edge_proposal, defect_types, roi_bytes, status, verdict, defect_type, reviewer, reviewed_at "
                          "FROM items WHERE status=? ORDER BY received DESC LIMIT ?", (status, limit)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/roi/{item_id}")
def roi(item_id: str):
    with lock:
        r = db.execute("SELECT roi FROM items WHERE id=?", (item_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    return Response(r["roi"], media_type="image/jpeg")


class Review(BaseModel):
    verdict: str
    defect_type: str | None = None
    reviewer: str = "reviewer"
    note: str | None = None


@app.post("/api/review/{item_id}")
def review(item_id: str, rv: Review):
    if rv.verdict not in ("good", "defective"):
        raise HTTPException(400, "verdict must be good or defective")
    with lock:
        db.execute("UPDATE items SET status='reviewed', verdict=?, defect_type=?, reviewer=?, reviewed_at=?, note=? WHERE id=?",
                   (rv.verdict, rv.defect_type, rv.reviewer, time.time(), rv.note, item_id))
        db.commit()
    return {"ok": True}


@app.get("/api/labels")
def labels(site: str, since: float = 0.0, x_nanoinspect_key: str = Header(None)):
    _auth(x_nanoinspect_key)
    with lock:
        rows = db.execute("SELECT id, verdict, defect_type, reviewer, reviewed_at FROM items WHERE site=? AND status='reviewed' "
                          "AND reviewed_at>? ORDER BY reviewed_at", (site, since)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/fleet")
def fleet():
    with lock:
        sites = [dict(r) for r in db.execute(
            "SELECT site, COUNT(*) AS received, SUM(status='reviewed') AS reviewed, SUM(status='pending') AS pending, "
            "SUM(roi_bytes) AS bytes, SUM(verdict='defective') AS confirmed_defects FROM items GROUP BY site")]
        rev = db.execute("SELECT received, reviewed_at, verdict, vlm_verdict, bucket, category, defect_type FROM items WHERE status='reviewed'").fetchall()
    ttr = [r["reviewed_at"] - r["received"] for r in rev]
    agree = [r for r in rev if r["vlm_verdict"] in ("good", "defective")]
    buckets, types = {}, {}
    for r in rev:
        buckets.setdefault(r["bucket"], {"reviewed": 0, "defective": 0})
        buckets[r["bucket"]]["reviewed"] += 1; buckets[r["bucket"]]["defective"] += r["verdict"] == "defective"
        if r["verdict"] == "defective":
            k = f"{r['category']}/{r['defect_type'] or 'unspecified'}"; types[k] = types.get(k, 0) + 1
    return {"sites": sites, "reviewed": len(rev),
            "median_minutes_to_review": statistics.median(ttr) / 60 if ttr else None,
            "vlm_agreement_with_human": (sum(r["vlm_verdict"] == r["verdict"] for r in agree) / len(agree)) if agree else None,
            "by_bucket": buckets, "defect_library": dict(sorted(types.items(), key=lambda kv: -kv[1]))}


@app.post("/api/reset")
def reset(x_nanoinspect_key: str = Header(None)):
    _auth(x_nanoinspect_key)
    with lock:
        db.execute("DELETE FROM items"); db.commit()
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
