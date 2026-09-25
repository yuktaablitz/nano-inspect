"""Cost-based escalation policy.

After the two edge LLMs have looked at a part, it falls into one of a few *evidence buckets*
(both say good, both say defective, they disagree, ...). For each bucket we measure how often it
occurs for defective and for good parts: P(bucket | defect) and P(bucket | good). With the line's
defect rate as the prior, Bayes' rule gives P(defect | bucket). The automatic decision then has an
expected cost of
    accept: P(defect | bucket) x cost of an escaped defect
    reject: P(good   | bucket) x cost of scrapping a good part
and we escalate to a human in the cloud only when the cheaper automatic option still costs more
than a human review. The decision is explicit (a table), defensible (the business's own costs and
measured error rates), and measurable (the escalation rate and total cost on held-out data).
"""
import json

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

BUCKETS = {
    "fast_accept": "tier 1 (fine-tuned 7B) is confident the part is good; accepted without tier 2 (shown as escape risk)",
    "agree_good": "tier 1 and tier 2 both see a good part",
    "agree_defect": "tier 1 and tier 2 both see a defect",
    "t1_flag_t2_good": "tier 1 flags the part, tier 2 (27B) sees no defect",
    "t1_good_t2_defect": "tier 1 says good, tier 2 sees a defect (audit sample)",
    "t2_unusable": "tier 1 flags the part, tier 2 answer unusable",
}
DEFAULT_COSTS = {"escape_usd": 50.0, "false_reject_usd": 2.0, "human_review_usd": 0.50}
DEFAULT_DEFECT_RATE = 0.05
POLICY_PATH = RESULTS_DIR / "escalation_policy.json"


def bucket(score, review_at, verdict, vlm_consulted=True):
    if not vlm_consulted:
        return "fast_accept" if score < review_at else "t2_unusable"
    if verdict not in ("good", "defective"):
        return "t2_unusable"
    if score < review_at:
        return "agree_good" if verdict == "good" else "t1_good_t2_defect"
    return "agree_defect" if verdict == "defective" else "t1_flag_t2_good"


def fit_likelihoods(df):
    """df: label, score, review_at, verdict (fine-tuned VLM). Every part is assigned to the bucket
    it would reach if the VLM were consulted; 'fast_accept' uses the vision score alone."""
    b = [bucket(r.score, r.review_at, r.verdict) for r in df.itertuples()]
    fast = df.score < df.review_at
    lik = {}
    for name in BUCKETS:
        if name == "fast_accept":
            d, g = fast[df.label == 1], fast[df.label == 0]
            lik[name] = {"p_given_defect": (d.sum() + 1) / (len(d) + 2), "p_given_good": (g.sum() + 1) / (len(g) + 2),
                         "n_defect": int(d.sum()), "n_good": int(g.sum())}
        else:
            m = np.array(b) == name
            d, g = m[df.label.values == 1], m[df.label.values == 0]
            lik[name] = {"p_given_defect": (d.sum() + 1) / (len(d) + 2), "p_given_good": (g.sum() + 1) / (len(g) + 2),
                         "n_defect": int(d.sum()), "n_good": int(g.sum())}
    return lik


def policy_table(lik, costs=DEFAULT_COSTS, defect_rate=DEFAULT_DEFECT_RATE):
    rows = []
    for name, l in lik.items():
        pd_ = defect_rate * l["p_given_defect"]; pg = (1 - defect_rate) * l["p_given_good"]
        p = pd_ / (pd_ + pg)
        c_acc, c_rej = p * costs["escape_usd"], (1 - p) * costs["false_reject_usd"]
        auto = "accept" if c_acc <= c_rej else "reject"
        c_auto = min(c_acc, c_rej)
        action = "escalate_to_human" if c_auto > costs["human_review_usd"] else auto
        rows.append({"bucket": name, "meaning": BUCKETS[name], "p_defect": p, "expected_cost_accept_usd": c_acc,
                     "expected_cost_reject_usd": c_rej, "human_review_usd": costs["human_review_usd"], "action": action,
                     "evidence_defect_parts": l["n_defect"], "evidence_good_parts": l["n_good"]})
    return pd.DataFrame(rows)


def simulate(df, table, defect_rate, costs, audit_rate=0.05):
    """Apply the cascade to held-out parts, reweighted to the given defect rate.
    Returns rates per 1,000 parts: escapes, false rejects, VLM calls, human reviews, and cost."""
    act = table.set_index("bucket").action.to_dict()
    w = np.where(df.label == 1, defect_rate / (df.label == 1).mean(), (1 - defect_rate) / (df.label == 0).mean())
    w = w / w.sum() * 1000
    esc = fr = vlm = human = 0.0
    for r, wi in zip(df.itertuples(), w):
        flagged = r.score >= r.review_at
        # Tier 1: unflagged parts are accepted, except the audit sample that also goes to the VLM
        if not flagged:
            # Tier 1 accepts it; an audit sample is re-checked by the VLM and follows its bucket's action
            vlm += wi * audit_rate
            a = act[bucket(r.score, r.review_at, r.verdict)]
            if r.label == 1: esc += wi * (1 - audit_rate)
            if a == "escalate_to_human": human += wi * audit_rate
            elif a == "accept" and r.label == 1: esc += wi * audit_rate
            elif a == "reject" and r.label == 0: fr += wi * audit_rate
            continue
        vlm += wi
        a = act[bucket(r.score, r.review_at, r.verdict)]
        if a == "escalate_to_human": human += wi
        elif a == "accept" and r.label == 1: esc += wi
        elif a == "reject" and r.label == 0: fr += wi
    cost = esc * costs["escape_usd"] + fr * costs["false_reject_usd"] + human * costs["human_review_usd"]
    return {"escapes_per_1000": esc, "false_rejects_per_1000": fr, "vlm_calls_per_1000": vlm,
            "human_reviews_per_1000": human, "cost_per_1000_usd": cost}


def single_model(df, flagged, name, defect_rate, costs):
    """A single model decides alone: reject if it flags the part, accept otherwise."""
    y = df.label.values; f = np.asarray(flagged, bool)
    esc = defect_rate * 1000 * (~f[y == 1]).mean(); fr = (1 - defect_rate) * 1000 * f[y == 0].mean()
    return {name: {"escapes_per_1000": esc, "false_rejects_per_1000": fr, "vlm_calls_per_1000": 0.0, "human_reviews_per_1000": 0.0,
                   "cost_per_1000_usd": esc * costs["escape_usd"] + fr * costs["false_reject_usd"]}}


def baselines(df, defect_rate, costs):
    """Cost per 1,000 parts for simpler strategies on the same held-out parts."""
    return {"Human inspects every part": {"escapes_per_1000": 0.0, "false_rejects_per_1000": 0.0, "vlm_calls_per_1000": 0.0,
                                          "human_reviews_per_1000": 1000.0, "cost_per_1000_usd": 1000 * costs["human_review_usd"]},
            **single_model(df, df.score >= 0.5, "Tier 1 alone (fine-tuned 7B decides)", defect_rate, costs)}


def fit_and_save(df, seed=0, costs=DEFAULT_COSTS, defect_rate=DEFAULT_DEFECT_RATE, audit_rate=0.05, extra_baselines=None, t_lo=None):
    """df: label, score (tier-1 P(defective)), review_at (tier-1 fast-accept threshold), verdict (tier-2 verdict).
    Fit on one half of the evaluation parts, report on the other half (no tuning on the reported half)."""
    rng = np.random.default_rng(seed); fit_mask = rng.random(len(df)) < 0.5
    fit_df, test_df = df[fit_mask], df[~fit_mask]
    table = policy_table(fit_likelihoods(fit_df), costs, defect_rate)
    result = {"NanoInspect cascade": simulate(test_df, table, defect_rate, costs, audit_rate), **baselines(test_df, defect_rate, costs)}
    for name, flag_col in (extra_baselines or {}).items():
        result.update(single_model(test_df, test_df[flag_col].values, name, defect_rate, costs))
    lik_all = fit_likelihoods(df)          # the deployed policy uses all evaluation parts
    POLICY_PATH.write_text(json.dumps({"likelihoods": lik_all, "costs": costs, "defect_rate": defect_rate, "audit_rate": audit_rate, "t_lo": t_lo,
                                       "held_out_result": result, "fit_parts": int(fit_mask.sum()),
                                       "test_parts": int((~fit_mask).sum())}, indent=1, default=float))
    return table, pd.DataFrame(result).T, lik_all


def load():
    if POLICY_PATH.exists():
        return json.loads(POLICY_PATH.read_text())
    return None
