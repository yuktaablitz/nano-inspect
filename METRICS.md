# Metrics: what we measure, how, and why

Every number is produced on the HP ZGX Nano (NVIDIA GB10, 128 GB unified memory) by `nanoinspect.ipynb`.
Raw outputs are in `artifacts/results/`; the summary is `artifacts/results/benchmark_summary.json`.

## Principles
1. **Held-out only.** Real defects exist only in MVTec AD's test set. Each defect type is split 50/50 (seeded): split A may be used for training (the tier-1 LoRA), and split B plus every good test image is used only for evaluation. A content-hash check proves no evaluation file is in any training split.
2. **Same images for every model.** All LLM tiers and all baselines are scored on the same 1,096 images.
3. **Measure the decision, not just the model.** A plant pays for escaped defects, scrapped good parts and reviewers' time, so we report those per 1,000 parts.
4. **No tuning on reported numbers.** The tier-1 fast-accept threshold comes from validation good parts. The escalation policy is fitted on one half of the evaluation images and reported on the other.
5. **Measure on the device, as served.** Model latency and throughput are measured through the same vLLM servers the app uses.

## Model quality
| Metric | Why | How |
|---|---|---|
| ROC-AUC | Threshold-free ranking quality; comparable between LLMs and classical baselines | P(defective) for the LLMs = probability mass on "defective" vs "good" at the verdict token (vLLM `top_logprobs`); baselines use their own scores |
| Recall, false-positive rate, accuracy | What the line experiences at the model's own verdict | JSON verdict vs label |
| Defect-type accuracy, location accuracy | Operators need *what* and *where*; this is the LLMs' edge over score-only methods | Among detected defects: type vs folder name; 3×3 region vs ground-truth mask centroid |
| Valid-JSON rate | Automation breaks on malformed output | Share of answers that parse into an allowed verdict |
| Value of fine-tuning | Shows what on-device training buys | Same prompts and images: Qwen2.5-VL-7B zero-shot vs + LoRA |

## Serving performance (the model metrics after serving)
| Metric | Why | How |
|---|---|---|
| Requests/s and output tokens/s at 1–32 concurrent clients | Capacity per tier, and how many lines one Nano can serve | Concurrent clients against the vLLM OpenAI API |
| p50 / p95 latency | The line's time budget; the tails matter | Client-side, per request |
| Time to first token, time per output token, KV-cache use, prefix-cache hit rate | Server-side view of where time goes | vLLM Prometheus `/metrics`, shown live in the app |

## The escalation decision
| Metric | Why | How |
|---|---|---|
| P(defect \| evidence bucket) | Makes the rule explicit and auditable | Bayes' rule from P(bucket \| defect), P(bucket \| good) (Laplace-smoothed) and the line's defect rate |
| Escapes, false rejects, tier-2 calls, human reviews per 1,000 parts | What a plant manager budgets for | Cascade simulated on the held-out half, reweighted to the defect rate |
| Cost per 1,000 parts vs baselines | When the edge-first design pays off | vs "a human checks every part", tier 1 alone, tier 2 alone, ResNet alone, 27B zero-shot alone |
| Sensitivity | The decision holds across businesses, not just one set of numbers | Defect rate 0.5–20%, escape cost $10–$1,000 |

## Edge stress and safety
| Metric | Why | How |
|---|---|---|
| 15-min soak with both tiers under load | Sustained work, not a demo run; thermal throttling | 16 tier-1 clients + 6 tier-2 clients continuously; nvidia-smi every 2 s (temperature, SM clock, power, throttle flags) |
| Energy per inspection per tier | Edge economics | GPU power (nvidia-smi) × time ÷ inspections, idle reported separately; wall power is higher |
| Robustness | Cameras drift | Tier-1 ROC-AUC under brightness, blur, JPEG and rotation changes (300 stratified images) |
| Offline proof | "Runs at the edge" must hold with no network | Non-local sockets and DNS blocked in-process; full inspections run; connection attempts counted (0 expected) |
| Failure recovery | A broken camera or a crashed model must never produce "accept" | Corrupt, truncated, blank and noisy frames; a model server down; count accepts of broken inputs (0 expected) |
| Delta score (training-free) | An explicit, explainable good-vs-part distance | Nearest-neighbour patch distance to ~40 good parts per product, in σ above held-out good parts |
