# NanoInspect

**Every part inspected on the line. Only the hard cases leave the building.**

NanoInspect is an edge-first visual quality-inspection system. It runs a cascade of **two locally served vision-language models** on one **HP ZGX Nano** (NVIDIA GB10), fine-tunes the cheaper one on the device, and escalates to a human in the cloud only when a cost rule says a review is worth it. When a part is rejected, it sends a machine-readable, SOP-grounded instruction to the line. HP Edge AI SJSUHack, September 2026.

![architecture](docs/architecture.svg)

## Hackathon deliverables
| Deliverable | Where |
|---|---|
| **Public Git repository** with all source code | this repo. Model weights and the dataset are downloaded by `setup.sh`; the trained adapters are in the [`adapters-v1` release](https://github.com/yuktaablitz/nano-inspect/releases/tag/adapters-v1) |
| **README with setup instructions** | [Quick start](#quick-start) and [Run it](#run-it-on-a-zgx-nano-dgx-spark-or-any-nvidia-gb10-machine) |
| **How local / hybrid inference is implemented** | [How local / hybrid inference works](#how-local--hybrid-inference-works), [`docs/architecture.svg`](docs/architecture.svg) |
| **Script to automate the prototype** | [`setup.sh`](setup.sh) installs everything; [`start_all.sh`](start_all.sh) starts the whole prototype, checks it end to end, reports status and stops it ([details](#prototype-automation-script-start_allsh)) |
| **Interactive deck** | [`docs/presentation/NanoInspect_Presentation.html`](docs/presentation/NanoInspect_Presentation.html): 27 slides, self-contained, works offline. Download and open it in a browser; N = speaker notes, A = all slides |
| **Demo video (≤ 2 min)** | script, live-demo runbook and an animation prompt in [`docs/DEMO_AND_VIDEO.md`](docs/DEMO_AND_VIDEO.md) |

## Quick start
```bash
git clone https://github.com/yuktaablitz/nano-inspect.git && cd nano-inspect
./setup.sh                    # once: Python env, packages, model weights, trained adapters (then it works offline)
./start_all.sh --https        # start both AI tiers, the cloud review tier and the operator app; waits until ready
./start_all.sh demo           # automated check: a good part, a defect and a disagreement, each must PASS
```
Then open `https://<nano-ip>:8080`.

## Who it is for, and why the cloud alone does not work
**Target user:** the QC line supervisor at a manufacturer launching a new product, who has about one second per part to decide: ship it, scrap it, or check it again.
- The product's images *are* its design, and the company does not want them in a vendor's cloud yet (**data residency**).
- A cloud round-trip eats the one-second budget (**latency**), and the line cannot stop when the WAN drops (**connectivity**).
- Cloud inference and human inspection both cost money for every part (**per-call cost**).
- Today's alternatives: human inspectors catch about 80–85% of defects and can reject a third of good parts ([Sandia, 2015](https://www.osti.gov/servlets/purl/1236203)); AQL sampling inspects only a sample of each lot; rule-based vision needs an engineer per product. Details in [docs/QA_FOR_JUDGES.md](docs/QA_FOR_JUDGES.md).

## The system
| Tier | Runs on | Model | Sees |
|---|---|---|---|
| 1 · cheap | ZGX Nano, **HP Z Runtime** (vLLM) | **Qwen2.5-VL-7B + LoRA fine-tuned on the Nano** (0.48% of weights, 28 min, real defect photos only) | every part: verdict, defect type, location, P(defect) from token log-probabilities |
| 2 · expensive | ZGX Nano, **HP Z Runtime** (vLLM) | **Qwen3.8-27B + LoRA fine-tuned on the Nano** (79.7M of 27.4B parameters, 0.29%, 2 h 14 min), chosen from the [vLLM recipes](https://recipes.vllm.ai) verified for DGX Spark / GB10; served with **FP8 weights** + adapter (quantised at load: same accuracy as BF16, 1.8× faster; `artifacts/results/tier2_fp8_vs_bf16.json`), or untrained NVFP4 (`T2=nvfp4`) | parts tier 1 is unsure about, plus a 5% audit of accepts; compares with a known-good reference, explains, writes the NCR, answers operator chat |
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
- **Early decision:** tier 2 stops as soon as it has written verdict, defect type and location, which is all the decision needs. The operator sees a plain sentence built from those fields ("Visible broken large at the bottom-right…"). With direct vLLM (`SERVE=vllm`), the untrained 27B writes a free-text sentence in the background instead.
- **Operator chat** with tier 2 about any inspected part, streamed word by word (first words in under a second).
- **Your own SOP:** download the SOP in use, or upload the plant's `sop.json`. It is validated with plain-English errors, the previous version is kept in `config/sop_history/`, and it applies to the next decision.

## Baselines
Classical ResNet-18 classifier · training-free delta · zero-shot Qwen2.5-VL-7B · zero-shot Qwen3.8-27B · a human checks every part · tier 1 alone · tier 2 alone. All are evaluated on the same 1,096 held-out MVTec AD images (15 products, 73 real defect types).

## Results
All numbers are measured on the ZGX Nano, on the same 1,096 held-out images (629 defective, 467 good) that no model trained on.

**Every model against the baselines** (tier 2 before → after its LoRA fine-tune on the Nano):

| Model | ROC-AUC | Defects caught | Good parts flagged | Right defect type | Right location |
|---|---|---|---|---|---|
| **Tier 1: Qwen2.5-VL-7B + LoRA (ours)** | **0.962** | **84.4%** | **5.6%** | **73.6%** | **70.6%** |
| **Tier 2: Qwen3.8-27B + LoRA + good reference (ours)** | 0.956 → **0.988** | 89.2% → **94.3%** | 11.1% → **4.7%** | 54.0% → **78.1%** | 65.6% → **80.8%** |
| Baseline: Qwen2.5-VL-7B zero-shot | 0.837 | 20.8% | 1.7% | 52.7% | 48.1% |
| Baseline: Qwen3.8-27B zero-shot, single image | 0.924 | 74.7% | 7.7% | 51.3% | 64.3% |
| Baseline: ResNet-18 classifier (not an LLM) | 0.960 | 92.8% | 14.8% | – | – |
| Baseline: training-free difference map (not an LLM) | 0.892 | 62.5% | 4.1% | – | – |

**Whole strategies, per 1,000 parts** (held-out half; $50 per escaped defect, $2 per scrapped good part, $0.50 per human review; 5% defect rate):

| Strategy | Cost | Escaped defects | Good parts scrapped | Human reviews | Tier-2 calls |
|---|---|---|---|---|---|
| **NanoInspect, capacity mode** (tier 2 on as many parts as it can serve) | **$230** | 3.9 | 0 | 73 | 524 |
| Tier 2 alone (fine-tuned 27B decides every part) | $217 | 3.0 | 34.1 | 0 | 1,000 (too slow for the line: 0.46 parts/s) |
| **NanoInspect, throughput mode (default)** | **$412** | 7.2 | 11.4 | 56 | 151 |
| ResNet-18 alone | $445 | 3.8 | 128.7 | 0 | – |
| A person checks every part | $500 | 0 | 0 | 1,000 | – |
| Tier 1 alone | $505 | 7.5 | 64.3 | 0 | – |
| 27B zero-shot alone | $778 | 12.5 | 75.7 | 0 | – |

**Tier-2 fine-tune decision:** the rule set before training was to keep it only if the cascade cost per 1,000 parts beat the baseline's $415.24. It came in at $411.91, so we **kept** it (`02_tier2_finetune_and_capacity.ipynb`; the baseline is tagged `pre-tier2-finetune`). The price: the adapter needs the BF16 weights, so tier 2 serves 0.46 parts/s instead of 0.95 with NVFP4, and uses 122 J per call instead of 51.

**Tier 2 in FP8 vs BF16** (the same 300 held-out images, 150 defective and 150 good): defects caught 95.3% vs 95.3%, good parts flagged 5.3% vs 4.7%, ROC-AUC 0.990 vs 0.992, same verdict on 99% of images; 0.84 vs 0.46 parts/s. FP8 is the default.

**Tier 2 through HP Z Runtime** (the same 300 images): defects caught 95.3%, good parts flagged 4.0%, ROC-AUC 0.991, the same verdict on 99.7% of images, same throughput (`artifacts/results/tier2_zrt_vs_bf16.json`).

**Time to a decision on the Nano** (live app, `./start_all.sh demo`):

| Part | Outcome | Before optimising | Now |
|---|---|---|---|
| Good part | accept at tier 1 | 2.8 s | **1.9 s** |
| Clear defect | reject at tier 2 + machine instruction | 45.0 s | **6.4 s** |
| Models disagree | human review | 19.8 s | **9.0 s** |

Three changes did it: FP8 weights for tier 2 (half the memory read per generated token), the early decision, and reusing tier 2's sentence for the defect report instead of a separate 27B call.

**Edge performance (fine-tuned setup):**

| Metric | Result |
|---|---|
| Soak test, 15 min, both tiers | 2,671 + 180 inferences, max 70 °C, no throttling, 0 errors |
| Energy per inspection (GPU) | tier 1 13.7 J · tier 2 122.2 J |
| Offline (network blocked) | 3/3 full inspections, 0 outbound connections |
| Failure cases | 9 tested, 0 broken inputs accepted |
| Cloud API cost avoided (60 parts/min, 16 h) | ≈ $89/day, 30M tokens, 66,000 calls (GPT-4o-equivalent rates) |

See [METRICS.md](METRICS.md) for how and why each metric was chosen. The full numbers are in `artifacts/results/benchmark_summary.json`.

## How local / hybrid inference works
NanoInspect is **local-first, hybrid by exception**: every model call happens on the ZGX Nano, and the cloud only stores what a human needs to see.

```
camera / upload ─▶ input check ─▶ TIER 1 (7B + LoRA, via HP Z Runtime) ──P(defect) < T_LO──▶ ACCEPT          (≈1–2 s, most parts)
   (unreadable or out-of-distribution frames go straight to a human)   │ unsure, flagged, or 5% audit
                                                                        ▼
                                  TIER 2 (27B + LoRA + known-good reference, via HP Z Runtime)
                                                                        │ evidence bucket ─▶ P(defect | bucket) (Bayes)
                                                                        ▼
                       min(P·$escape, (1−P)·$scrap) ≤ $review ?  ── yes ─▶ ACCEPT or REJECT on the edge
                                                                        │ no: the models disagree, or tier 2 could not answer
                                                                        ▼
               CLOUD: store-and-forward outbox (SQLite) ─▶ human review console :9000 ─▶ label back to the edge
```

- **Local (edge):** both vision-language models are served on the Nano by **HP Z Runtime (`zrt`)**, HP's serving layer, which runs vLLM underneath. `scripts/zrt_serve.sh` starts each tier with `zrt service start` and our exact vLLM settings, and both tiers sit behind ZRT's HTTPS proxy on `127.0.0.1:8100`. Each ZRT service is labelled with its fine-tuned adapter's name (`nanoinspect-7b-lora`, `nanoinspect-27b-lora`), because the proxy routes requests by label. The app talks to it through an OpenAI-compatible API (`nanoinspect/serving.py`). The per-part cascade is in `nanoinspect/cascade.py` (`decide_part`), and the escalation rule is in `nanoinspect/policy.py`. A rejected part gets its SOP-grounded machine JSON (`nanoinspect/actions.py`) on the edge, so the line keeps running with the WAN down.
- **Tier 1 → tier 2:** tier 2 runs only when tier 1's P(defect) (from verdict-token log-probabilities) is at or above `T_LO`, plus a 5% random audit of accepts. Default is *throughput mode* (`T_LO` 0.5). *Capacity mode* (`T_LO` 0.047) sends as many parts to tier 2 as it can serve; switch in the app under Policy.
- **Tier 2 → cloud:** a part leaves the building only when the two tiers disagree (tier 1 flags, tier 2 clears it, or the reverse), tier 2's answer is unusable, or the input itself is unreadable, *and* the cost rule says a human review is cheaper than the risk. The verdict, the explanation and the operator chat all come from the fine-tuned 27B.
- **Hybrid (cloud):** `nanoinspect/escalation.py` writes each escalation to a local SQLite outbox (`artifacts/edge.db`) and forwards only a crop plus the model evidence to `cloud/server.py`, retrying through outages. **The cloud runs no AI.** A reviewer's label returns to the edge as training data.
- **Offline proof:** the notebook blocks every outbound connection in-process and runs full inspections (3/3 done, 0 connection attempts).

## Run it (on a ZGX Nano, DGX Spark, or any NVIDIA GB10 machine)
Needs Ubuntu with an NVIDIA driver and CUDA 13, Python 3.12, about 100 GB of disk for weights, and the MVTec AD dataset.
```bash
git clone https://github.com/yuktaablitz/nano-inspect.git && cd nano-inspect
./setup.sh              # 1. venv + packages, model weights, trained LoRA adapters from the GitHub release (then works offline)
export NANOINSPECT_DATA=$HOME/Downloads/mvtec_anomaly_detection   # 2. MVTec AD (CC BY-NC-SA 4.0), download once
./start_all.sh          # 3. ONE COMMAND: tier 1 + fine-tuned tier 2 (vLLM), cloud review tier, edge app; waits until ready
./start_all.sh demo     # 4. automated end-to-end check: runs accept / reject / human-review parts and verifies each outcome
```

### Prototype automation script (`start_all.sh`)
| Command | What it does |
|---|---|
| `./start_all.sh` | Starts every component that isn't already running, waits for each to be healthy, and prints the URLs |
| `./start_all.sh --https` | The same, with the operator app on HTTPS across the network, so phones and laptops can use the live camera. It creates a self-signed certificate for the Nano's IP addresses |
| `./start_all.sh demo` | Runs the three demo parts through the live pipeline and prints PASS/FAIL, tier and time for each. Exits non-zero on failure, so it works in CI |
| `./start_all.sh status` | Health of tier 1, tier 2, the cloud tier and the edge app |
| `./start_all.sh stop` | Stops what the script started |
| `TIER2_QUANT= ./start_all.sh` | Serves tier 2 in full BF16 instead of FP8 |
| `T2=nvfp4 ./start_all.sh` | Serves the untrained NVFP4 27B instead of the fine-tuned model |
| `SERVE=vllm ./start_all.sh` | Serves both tiers with vLLM directly (`serve_models.sh`) instead of HP Z Runtime, e.g. on a GB10 machine without ZRT |
| `scripts/zrt_serve.sh setup · pull · start · status · stop` | HP Z Runtime only: configure the proxy (port 8100, TLS), download the models into ZRT's cache, start or stop both tiers |

Example `demo` output on the ZGX Nano:
```
  PASS  Good part          expected accept        got accept        at tier 1    1.9 s
  PASS  Clear defect       expected reject        got reject        at tier 2    6.4 s
  PASS  Models disagree    expected manual_review got manual_review at tier 3    9.0 s
All three decisions as expected.
```
- The first start compiles GB10 kernels (about 10 min). Later, loading the models takes about 10 minutes and the operator app about 5, so start well before a demo.
- First time with HP Z Runtime: `scripts/zrt_serve.sh setup` then `scripts/zrt_serve.sh pull` (downloads ~72 GB into ZRT's cache).
- Step by step instead: `scripts/zrt_serve.sh start` (or, without ZRT, `./serve_models.sh tier1` and `TIER2_QUANT=fp8 ./serve_models.sh tier2ft`), `./run_cloud.sh &`, `./run_edge.sh --https`. The cloud tier can also run elsewhere: `docker build -f cloud/Dockerfile -t nanoinspect-cloud .`, then set `NANOINSPECT_CLOUD_URL`.
- Reproduce every number: `jupyter nbconvert --to notebook --execute --inplace nanoinspect.ipynb`, then `02_tier2_finetune_and_capacity.ipynb`. Retrain the adapters with `python -m nanoinspect.vlm` (7B, 28 min) and `python -m nanoinspect.finetune_t2` (27B, 2 h 14 min).

From a laptop, either run `./start_all.sh --https` and open `https://<nano-ip>:8080`, or tunnel with `ssh -L 8080:localhost:8080 -L 9000:localhost:9000 <user>@<nano-ip>` and open http://localhost:8080.

The operator app has five tabs, and every page is reachable from exactly one of them:

| Tab | Pages |
|---|---|
| **Inspect** | upload a photo or take a picture, preview it, then **Submit for inspection**; samples and the three demo parts; chat with tier 2 about the result |
| **Operations** | Live overview · Production line (simulator + line controller) · Human review queue · Cloud review console ↗ |
| **Quality rules** | Escalation policy & costs · SOP & line instructions (view, download, upload) |
| **Performance** | Accuracy & fine-tuning · Cost savings · Models & serving · Stress tests |
| **How it works** | the pipeline, step by step |

- `/pitch`: interactive pitch with live numbers.
- **[docs/presentation/NanoInspect_Presentation.html](docs/presentation/NanoInspect_Presentation.html)**: the 20-minute interactive deck. It is self-contained and works offline: download it and open it in a browser. Press N for speaker notes and A to see all slides.
- http://localhost:9000: the cloud review console.

The trained LoRA adapters (tier 1: 149 MB, tier 2: 294 MB zipped) are too large for git. `setup.sh` downloads them from the [`adapters-v1` release](https://github.com/yuktaablitz/nano-inspect/releases/tag/adapters-v1).

## Repository
| Path | What |
|---|---|
| `02_tier2_finetune_and_capacity.ipynb` | Keep-or-revert decision for the tier-2 fine-tune; capacity-mode policy |
| `nanoinspect.ipynb` | Model choice, fine-tune record, quality vs baselines, serving benchmark, escalation policy, machine JSON, stress tests, line simulation |
| `nanoinspect/serving.py`, `cascade.py` | Clients for the vLLM tiers; the per-part cascade |
| `nanoinspect/vlm.py`, `finetune_t2.py` | LoRA fine-tuning of Qwen2.5-VL-7B (tier 1) and Qwen3.8-27B (tier 2) on the Nano |
| `nanoinspect/policy.py` | Cost-based escalation policy |
| `nanoinspect/delta.py` | Training-free delta map |
| `nanoinspect/actions.py`, `config/sop.json` | SOP-grounded machine instructions |
| `nanoinspect/escalation.py`, `cloud/` | Edge outbox and store-and-forward; cloud review service + Dockerfile |
| `nanoinspect/server.py`, `nanoinspect/web/` | Edge web app and pitch |
| `nanoinspect/stress_llm.py`, `linesim.py`, `telemetry.py` | Soak, energy, robustness, offline, failure tests; line simulator; GPU telemetry |
| `nanoinspect/vision.py`, `defects.py` | ResNet-18 baseline and its synthetic defects |
| `setup.sh`, `start_all.sh` | Install everything; start, check (`demo`), monitor (`status`) and stop the whole prototype |
| `docs/presentation/` | The interactive presentation (single HTML file) |
| `serve_models.sh`, `run_edge.sh`, `run_cloud.sh` | Start one part: model tiers, edge app, cloud tier |
| `docs/DEMO_AND_VIDEO.md` | Demo runbook, 2-minute video script, animation prompt |
| `docs/` | Architecture diagram, Q&A for judges, vLLM recipe snapshot |

## Rules compliance
All inference, training and fine-tuning run on the ZGX Nano with open-weight models (Qwen2.5-VL-7B-Instruct and Qwen3.8-27B / nvidia/Qwen3.8-27B-NVFP4: Apache 2.0). The cloud tier runs no AI. The offline proof blocks every outbound connection while full inspections run (notebook section 8).

## Limits
MVTec AD is a public benchmark, not a production line. The default costs and the SOP are illustrative, and the cloud tier runs on our own machine in the demo.
