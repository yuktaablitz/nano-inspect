"""From a verdict to machine-readable action: the line command, the nonconformance report (NCR) and
containment. Grounded in the plant's SOP file (config/sop.json); tier 2 only fills in text fields and
may only pick a cause the SOP lists. Every message is validated against DISPOSITION_SCHEMA before it
is sent to the line controller.
"""
import collections
import json
import re
import threading
import time
import uuid

import jsonschema

from .config import PROJECT_DIR

SOP_PATH = PROJECT_DIR / "config" / "sop.json"

DISPOSITION_SCHEMA = {
    "type": "object",
    "required": ["schema", "message_id", "timestamp", "station", "product", "part_id", "disposition", "line_command", "sop"],
    "properties": {
        "schema": {"const": "nanoinspect.disposition/v1"},
        "message_id": {"type": "string"}, "timestamp": {"type": "string"}, "station": {"type": "string"},
        "product": {"type": "string"}, "part_id": {"type": "string"},
        "disposition": {"enum": ["release", "reject", "hold_for_mrb", "hold_for_review"]},
        "line_command": {"type": "object", "required": ["target", "action"],
                         "properties": {"target": {"type": "string"},
                                        "action": {"enum": ["pass", "divert_to_reject_bin", "divert_to_hold_bin", "stop_line"]}}},
        "sop": {"type": "object", "required": ["sop_id", "version"]},
        "nonconformance": {"type": "object",
                           "required": ["ncr_id", "defect_type", "severity", "location", "description", "probable_cause",
                                        "process_check", "rework_allowed", "operator_instructions"],
                           "properties": {"severity": {"enum": ["critical", "major", "minor"]},
                                          "rework_allowed": {"type": "boolean"},
                                          "operator_instructions": {"type": "array", "items": {"type": "string"}, "maxItems": 5}}},
        "containment": {"type": "object"},
        "evidence": {"type": "object"},
    },
}


def load_sop():
    return json.loads(SOP_PATH.read_text())


SEVERITIES = ("critical", "major", "minor")
DISPOSITIONS = ("reject", "hold_for_mrb", "hold_for_review", "release")
LINE_ACTIONS = ("pass", "divert_to_reject_bin", "divert_to_hold_bin", "stop_line")


def validate_sop(d, known_products=()):
    """Checks an uploaded SOP before it can drive the line. Returns (errors, warnings); only an SOP with no errors is used."""
    err, warn = [], []
    if not isinstance(d, dict):
        return ["The file must contain one JSON object."], warn
    for k in ("sop_id", "version"):
        if not isinstance(d.get(k), str) or not d.get(k).strip():
            err.append(f'"{k}" is required (text).')
    disp = d.get("dispositions")
    if not isinstance(disp, dict) or not disp:
        err.append('"dispositions" is required: one rule per severity (critical / major / minor).'); disp = {}
    for sev, r in disp.items():
        if sev not in SEVERITIES: err.append(f'dispositions: unknown severity "{sev}" (use {", ".join(SEVERITIES)}).'); continue
        if not isinstance(r, dict): err.append(f"dispositions.{sev} must be an object."); continue
        if r.get("disposition") not in DISPOSITIONS: err.append(f"dispositions.{sev}.disposition must be one of {', '.join(DISPOSITIONS)}.")
        if r.get("line_command") not in LINE_ACTIONS: err.append(f"dispositions.{sev}.line_command must be one of {', '.join(LINE_ACTIONS)}.")
        c = r.get("containment")
        if not isinstance(c, dict): err.append(f"dispositions.{sev}.containment is required."); continue
        for k in ("stop_line_if_repeats", "window_parts", "recheck_last_n"):
            if not isinstance(c.get(k), int) or c.get(k) < 0: err.append(f"dispositions.{sev}.containment.{k} must be a whole number >= 0.")
        if not isinstance(c.get("notify"), list): err.append(f"dispositions.{sev}.containment.notify must be a list of roles.")
    prods = d.get("products")
    if not isinstance(prods, dict) or not prods:
        err.append('"products" is required: one entry per product with its station and defect types.'); prods = {}
    for cat, pr in prods.items():
        if not isinstance(pr, dict) or not isinstance(pr.get("defects"), dict) or not pr["defects"]:
            err.append(f"products.{cat} needs a non-empty \"defects\" object."); continue
        if not isinstance(pr.get("station"), str): err.append(f"products.{cat}.station is required (text).")
        for dt, e in pr["defects"].items():
            where = f"products.{cat}.defects.{dt}"
            if not isinstance(e, dict): err.append(f"{where} must be an object."); continue
            if e.get("severity") not in disp: err.append(f"{where}.severity must be one of the severities in dispositions ({', '.join(disp) or 'none'}).")
            if not isinstance(e.get("likely_causes"), list) or not e["likely_causes"] or not all(isinstance(x, str) for x in e["likely_causes"]):
                err.append(f"{where}.likely_causes must be a non-empty list of text.")
            if not isinstance(e.get("process_check"), str): err.append(f"{where}.process_check is required (text).")
            if not isinstance(e.get("rework_allowed"), bool): err.append(f"{where}.rework_allowed must be true or false.")
    missing = [c for c in known_products if c not in prods]
    if missing: warn.append(f"No SOP entry for {', '.join(missing)}: defects on these products will be held for a person.")
    return err[:25], warn


def save_sop(d):
    """Backs up the current SOP to config/sop_history/ and writes the new one. Returns the backup path."""
    hist = SOP_PATH.parent / "sop_history"; hist.mkdir(exist_ok=True)
    bak = hist / f"sop-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.json"
    bak.write_text(SOP_PATH.read_text())
    d = {"effective": time.strftime("%Y-%m-%d"), "owner": "Quality", "basis": "", "note": "",
         "accept": {"disposition": "release", "line_command": "pass"},
         "manual_review": {"disposition": "hold_for_review", "line_command": "divert_to_hold_bin"}, **d}
    SOP_PATH.write_text(json.dumps(d, indent=2))
    return bak, d


def ncr_prompt(cat, station, defect_type, entry, explanation):
    return (f"You are the quality engineer for the {station} station. A {cat.replace('_', ' ')} was rejected by visual "
            f"inspection. Detected defect: {defect_type.replace('_', ' ')}. Inspector note: {explanation or 'none'}.\n"
            f"Plant SOP lists these likely causes: {entry['likely_causes']}. Process check: {entry['process_check']}. "
            f"Rework allowed by SOP: {entry['rework_allowed']}.\n"
            "The FIRST image is a known-good reference, the SECOND is the rejected part. Write the nonconformance report as JSON only: "
            '{"description": "one sentence describing the defect as seen", "probable_cause": one of the listed likely causes or "unknown", '
            '"operator_instructions": ["at most 3 short imperative steps consistent with the SOP"]}')


class LineController:
    """Stands in for the PLC / MES endpoint (in a plant: OPC UA or MQTT). Keeps every message and applies
    the SOP's containment rule: stop the line if one defect type repeats too often within a window."""

    def __init__(self, maxlen=500):
        self.messages = collections.deque(maxlen=maxlen)
        self.recent = collections.defaultdict(lambda: collections.deque(maxlen=200))
        self.lock = threading.Lock()
        self.stopped = None

    def send(self, msg):
        jsonschema.validate(msg, DISPOSITION_SCHEMA)
        with self.lock:
            self.messages.appendleft(msg)
        return msg

    def check_containment(self, product, defect_type, containment, part_no):
        with self.lock:
            q = self.recent[(product, defect_type)]; q.append(part_no)
            window, limit = containment.get("window_parts", 50), containment.get("stop_line_if_repeats", 999)
            hits = [p for p in q if part_no - p < window]
            if len(hits) >= limit and not self.stopped:
                self.stopped = {"product": product, "defect_type": defect_type, "hits": len(hits), "window": window,
                                "time": time.strftime("%H:%M:%S")}
                return True
        return False


class ActionBuilder:
    def __init__(self, tier2=None, refs=None, controller=None, writer_model=None):
        self.sop = load_sop(); self.t2 = tier2; self.refs = refs or {}; self.writer_model = writer_model
        self.controller = controller or LineController(); self._n = 0; self.last_usage = (0, 0)

    def _entry(self, cat, defect_type):
        return self.sop["products"].get(cat, {}).get("defects", {}).get(defect_type)

    def build(self, *, cat, decision, part_id, defect_type=None, location=None, explanation=None, evidence=None, pil=None, use_llm=True):
        """Returns (message, source) where source says whether the NCR text came from tier 2 or the SOP template."""
        self._n += 1
        sop_ref = {"sop_id": self.sop["sop_id"], "version": self.sop["version"]}
        station = self.sop["products"].get(cat, {}).get("station", "inspection")
        base = {"schema": "nanoinspect.disposition/v1", "message_id": uuid.uuid4().hex[:12],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "station": station, "product": cat, "part_id": str(part_id),
                "sop": sop_ref, "evidence": evidence or {}}
        if decision == "accept":
            return self.controller.send({**base, "disposition": "release", "line_command": {"target": "diverter", "action": "pass"}}), "sop"
        if decision == "manual_review" or not defect_type or not self._entry(cat, defect_type):
            msg = {**base, "disposition": "hold_for_review", "line_command": {"target": "diverter", "action": "divert_to_hold_bin"}}
            return self.controller.send(msg), "sop"
        entry = self._entry(cat, defect_type); rule = self.sop["dispositions"][entry["severity"]]
        ncr = {"ncr_id": f"NCR-{time.strftime('%Y%m%d')}-{self._n:05d}", "defect_type": defect_type, "severity": entry["severity"],
               "location": location or "unknown", "description": explanation or f"{defect_type.replace('_', ' ')} detected",
               "probable_cause": entry["likely_causes"][0], "process_check": entry["process_check"], "rework_allowed": entry["rework_allowed"],
               "operator_instructions": [f"Place the part in the {'reject' if rule['disposition'] == 'reject' else 'hold'} bin and tag it with {defect_type.replace('_', ' ')}",
                                         entry["process_check"]]}
        source = "sop + tier-2 sentence" if explanation else "sop"
        if use_llm and self.t2 is not None and pil is not None and cat in self.refs:
            try:
                r = self.t2.session.post(f"{self.t2.url}/v1/chat/completions", timeout=90, json={
                    "model": self.writer_model or self.t2.model, "temperature": 0, "max_tokens": 220, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": _url(self.refs[cat])}}, {"type": "image_url", "image_url": {"url": _url(pil)}},
                        {"type": "text", "text": ncr_prompt(cat, station, defect_type, entry, explanation)}]}]}).json()
                u = r.get("usage", {}); self.last_usage = (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                m = re.search(r"\{.*\}", r["choices"][0]["message"]["content"], re.S)
                d = json.loads(m.group(0)) if m else {}
                if isinstance(d.get("description"), str): ncr["description"] = d["description"][:300]
                if d.get("probable_cause") in entry["likely_causes"] + ["unknown"]: ncr["probable_cause"] = d["probable_cause"]
                steps = [s for s in d.get("operator_instructions", []) if isinstance(s, str)][:3]
                if steps: ncr["operator_instructions"] = steps; source = "tier2"
            except Exception:
                source = "sop (tier 2 unavailable)"
        msg = {**base, "disposition": rule["disposition"], "line_command": {"target": "diverter", "action": rule["line_command"]},
               "nonconformance": ncr, "containment": rule["containment"]}
        msg = self.controller.send(msg)
        if self.controller.check_containment(cat, defect_type, rule["containment"], self._n):
            stop = {**base, "message_id": uuid.uuid4().hex[:12], "disposition": rule["disposition"],
                    "line_command": {"target": station, "action": "stop_line"},
                    "containment": {**rule["containment"], "reason": f"{defect_type} repeated {self.controller.stopped['hits']} times "
                                                                     f"within {rule['containment']['window_parts']} parts"}}
            self.controller.send(stop)
        return msg, source


def _url(pil):
    from .serving import to_data_url
    return to_data_url(pil)
