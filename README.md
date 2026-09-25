# NanoInspect

**Every part inspected on the line. Only the hard cases leave the building.**

NanoInspect is an edge-first visual quality-inspection system. It runs a cascade of **two locally served vision-language models** on one **HP ZGX Nano** (NVIDIA GB10), fine-tunes the cheaper one on the device, and escalates to a human in the cloud only when a cost rule says a review is worth it. When a part is rejected, it sends a machine-readable, SOP-grounded instruction to the line. HP Edge AI SJSUHack, September 2026.

![architecture](docs/architecture.svg)

## Who it is for, and why the cloud alone does not work
**Target user:** the QC line supervisor at a manufacturer launching a new product, who has about one second per part to decide: ship it, scrap it, or check it again.
- The product's images *are* its design, and the company does not want them in a vendor's cloud yet (**data residency**).
- A cloud round-trip eats the one-second budget (**latency**), and the line cannot stop when the WAN drops (**connectivity**).
- Cloud inference and human inspection both cost money for every part (**per-call cost**).
- Today's alternatives: human inspectors catch about 80–85% of defects and can reject a third of good parts ([Sandia, 2015](https://www.osti.gov/servlets/purl/1236203)); AQL sampling inspects only a sample of each lot; rule-based vision needs an engineer per product. Details in [docs/QA_FOR_JUDGES.md](docs/QA_FOR_JUDGES.md).

## The system
| Tier | Runs on | Model | Sees |
|---|---|---|---|
| 1 · cheap | ZGX Nano, vLLM :8001 | **Qwen2.5-VL-7B + LoRA fine-tuned on the Nano** (0.48% of weights, 28 min, real defect photos only) | every part: verdict, defect type, location, P(defect) from token log-probabilities |
| 2 · expensive | ZGX Nano, vLLM :8002 | **Qwen3.8-27B NVFP4**, chosen from the [vLLM recipes](https://recipes.vllm.ai) verified for DGX Spark / GB10 | parts tier 1 is unsure about, plus a 5% audit of accepts; compares with a known-good reference, explains, writes the NCR, answers operator chat |
| 3 · cloud | any server | a human reviewer (**no AI in the cloud**) | only parts where review is cheaper than the risk |

**Escalation rule (explicit, defensible, measurable).** Both tiers' answers put a part in an evidence bucket. Bayes' rule turns error rates measured on held-out parts, plus the line's defect rate, into P(defect | bucket). Then:
```
escalate  ⇔  min( P(defect) × $escaped_defect ,  P(good) × $scrapped_part )  >  $human_review
```
The rule is fitted on one half of the evaluation images and reported on the other. The costs are editable in the app.

**Also on the edge:**
- An explicit **delta map**: per-patch distance to ~40 known-good parts. It is training-free and gives a heatmap.
- An input check for broken camera frames.
- A **SOP library** (`config/sop.json`, ISO 9001 §8.7) and **machine JSON** for the PLC / MES: divert, NCR, containment, and stop-line when a defect repeats.
- A store-and-forward outbox: only a crop of an escalated part is sent, and it queues through outages.
- Human labels flow back as training data.

## Baselines
Classical ResNet-18 classifier · training-free delta · zero-shot Qwen2.5-VL-7B · zero-shot Qwen3.8-27B · a human checks every part · tier 1 alone · tier 2 alone. All are evaluated on the same 1,096 held-out MVTec AD images (15 products, 73 real defect types).

## Results
Measured on the ZGX Nano, before any fine-tuning of tier 2:

| Metric                                              | Result                                                                |
|:----------------------------------------------------|:----------------------------------------------------------------------|
| Products / defect types                             | 15 products, 73 real defect types (MVTec AD), held-out evaluation     |
| Tier 1 ROC-AUC · recall (fine-tuned 7B)             | 0.963 · 84.4%   (zero-shot 7B: 0.836 · 21.0%)                         |
| Tier 1 names the right defect type · location       | 74.2% · 70.1%                                                         |
| Tier 2 ROC-AUC · recall (27B + reference)           | 0.956 · 89.2%   (27B zero-shot: 0.913 · 78.5%)                        |
| Classical baseline ResNet-18 ROC-AUC · recall       | 0.960 · 92.8% (no defect type or location)                            |
| LoRA fine-tune on the Nano                          | 28 min, 0.484% of parameters                                          |
| Serving throughput (vLLM on the Nano)               | tier 1 14.66 img/s at 32 clients · tier 2 0.95 img/s at 8 clients     |
| Cost per 1,000 parts: cascade vs human inspects all | $415.24 vs $500.00                                                    |
| Per 1,000 parts (cascade)                           | 7.28 escapes · 11.35 scrapped · 57.0 human reviews · 148 tier-2 calls |
| Soak 15 min, both tiers                             | 3,295 + 399 inferences · max 64 C · throttling: False · errors: 0     |
| Energy per inspection (GPU)                         | tier 1 12.7 J · tier 2 51.5 J                                         |
| Offline (network blocked)                           | 3/3 full inspections, 0 outbound connection attempts                  |
| Failure cases                                       | 9 tested, 0 broken inputs accepted                                    |

On cost, the cascade comes to $415 per 1,000 parts, against $500 for "a human checks every part", $445 for ResNet alone, $486 for tier 2 alone and $497 for tier 1 alone (held-out half, $50 escape / $2 scrap / $0.50 review, 5% defect rate). The app's Overview shows tokens processed locally, cloud API cost avoided (GPT-4o-equivalent rates, as in the ZGX Console), measured electricity, and network time avoided.

See [METRICS.md](METRICS.md) for how and why each metric was chosen. The full numbers are in `artifacts/results/benchmark_summary.json`.

## Run it on a ZGX Nano
```bash
git clone <this repo> && cd nanoinspect && ./setup.sh     # venv, packages, model weights (then works offline)
export NANOINSPECT_DATA=$HOME/Downloads/mvtec_anomaly_detection   # MVTec AD, CC BY-NC-SA 4.0
./serve_models.sh both        # tier 1 :8001 (7B + LoRA), tier 2 :8002 (27B NVFP4); first start compiles GB10 kernels
./run_cloud.sh &              # cloud review tier :9000 (or: docker build -f cloud/Dockerfile -t nanoinspect-cloud .)
./run_edge.sh                 # operator app :8080
jupyter nbconvert --to notebook --execute --inplace nanoinspect.ipynb    # all evaluations, serving benchmark, stress tests
```
From a laptop: `ssh -L 8080:localhost:8080 -L 9000:localhost:9000 <user>@<nano-ip>`, then open
- http://localhost:8080: operator console (overview, inspect with upload or webcam + chat, production line + line controller, cloud escalations, models and serving metrics, escalation policy and cost, evidence)
- http://localhost:8080/pitch: interactive pitch with live numbers
- http://localhost:9000: cloud review console

The LoRA adapter (160 MB) is too large for git; publish it as a release or on Hugging Face and set `ADAPTER_REPO` for `setup.sh`, or retrain with `nanoinspect/vlm.py` (section 3 of the notebook).

## Repository
| Path | What |
|---|---|
| `nanoinspect.ipynb` | Model choice, fine-tune record, quality vs baselines, serving benchmark, escalation policy, machine JSON, stress tests, line simulation |
| `nanoinspect/serving.py`, `cascade.py` | Clients for the vLLM tiers; the per-part cascade |
| `nanoinspect/vlm.py` | LoRA fine-tuning of Qwen2.5-VL-7B on the Nano |
| `nanoinspect/policy.py` | Cost-based escalation policy |
| `nanoinspect/delta.py` | Training-free delta map |
| `nanoinspect/actions.py`, `config/sop.json` | SOP-grounded machine instructions |
| `nanoinspect/escalation.py`, `cloud/` | Edge outbox and store-and-forward; cloud review service + Dockerfile |
| `nanoinspect/server.py`, `nanoinspect/web/` | Edge web app and pitch |
| `nanoinspect/stress_llm.py`, `linesim.py`, `telemetry.py` | Soak, energy, robustness, offline, failure tests; line simulator; GPU telemetry |
| `nanoinspect/vision.py`, `defects.py` | ResNet-18 baseline and its synthetic defects |
| `serve_models.sh`, `run_edge.sh`, `run_cloud.sh`, `setup.sh` | Scripts |
| `docs/` | Architecture, Q&A for judges, demo and video script, vLLM recipe snapshot |

## Rules compliance
All inference, training and fine-tuning run on the ZGX Nano with open-weight models (Qwen2.5-VL-7B-Instruct: Apache 2.0; nvidia/Qwen3.8-27B-NVFP4: Apache 2.0). The cloud tier runs no AI. The offline proof blocks every outbound connection while full inspections run (notebook section 8).

## Limits
MVTec AD is a public benchmark, not a production line. The default costs and the SOP are illustrative, and the cloud tier runs on our own machine in the demo.
