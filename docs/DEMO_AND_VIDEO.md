# Demo runbook and 2-minute video script

## Before the demo (on the Nano)
1. Stop other GPU jobs (for example the vLLM server on port 8000): `nvidia-smi` should list no other compute processes.
2. Terminal 1: `./run_cloud.sh` (cloud tier on :9000). Terminal 2: `./run_edge.sh` (edge app on :8080; it takes about 2 min to load the 7B VLM).
3. On the presenting laptop: `ssh -L 8080:localhost:8080 -L 9000:localhost:9000 hp14@<nano-tailscale-ip>`.
4. Open three tabs: `http://localhost:8080/pitch`, `http://localhost:8080`, `http://localhost:9000`.
5. Optional reset for a clean counter: `curl -X POST localhost:8080/api/reset` and `curl -X POST -H "X-NanoInspect-Key: nanoinspect-demo-key" localhost:9000/api/reset`.

## 5-minute live demo
| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Pitch, slides 1–3 | → | The user, the one-second budget, and three tiers where the cloud comes last |
| 0:45 | Console › Inspect | Product *screw*, "Random good part" | Tier 1 accepts in milliseconds; the VLM is not needed |
| 1:15 | Inspect | "Random defective part" | The fine-tuned VLM names the defect type and location; Grad-CAM shows where |
| 1:45 | Escalation policy | Set the escape cost from 50 to 500, Apply | The table changes: which evidence now justifies a human; cost per 1,000 parts vs baselines |
| 2:30 | Production line | Start at 240 parts/min | Every part at tier 1; a few go to the VLM; the fast path stays in milliseconds |
| 3:00 | Header | Switch the network OFF | The line keeps running; escalations queue on the Nano |
| 3:30 | Escalations, then network ON, then the cloud tab | Label one case with **D** | Crops arrive (KB, not MB); the label returns to the Nano as training data |
| 4:15 | Evidence (or pitch slides 6–9) | → | Zero-shot vs fine-tuned VLM, 15-min soak, energy, offline proof, cost per 1,000 parts |

**Backup:** record this run with OBS or QuickTime beforehand, in case the network between the pitch room and the Nano fails.

## 2-minute video (YouTube, public)
Put the team name, tagline and logo on screen at the start and the end.

| Time | Visual | Voice-over |
|---|---|---|
| 0:00–0:12 | Logo + tagline on dark background | "NanoInspect: every part inspected on the line, only the hard cases leave the building." |
| 0:12–0:30 | Slide 2 (target user) | "A line supervisor has one second per part. Cloud AI sends every image off-site and stops when the network drops. Human inspection doesn't scale." |
| 0:30–0:50 | Architecture diagram, highlight each tier | "On one HP ZGX Nano, a vision model checks every part in milliseconds. A 7-billion-parameter vision-language model, fine-tuned on the Nano, explains anything suspicious. A person in the cloud sees a part only when that is cheaper than the risk of a wrong call." |
| 0:50–1:25 | Screen recording: good part → defective part → line running → network off → cloud console label | "Here a good part is accepted instantly. A defective one: the VLM names a crack and where it is. We run the line and pull the network: inspection continues, and escalations wait on the device. Back online, only a small crop reaches the reviewer, and the label flows back to train the next model." |
| 1:25–1:50 | Evidence slide numbers | "Fine-tuning on the Nano took defect recall from X to Y. Fifteen minutes at full load with both models, no throttling. Zero network connections needed." |
| 1:50–2:00 | Impact slide + logo | "Lower cost per thousand parts, and product images stay in the building. NanoInspect, edge first, cloud only when it pays." |

Fill X and Y from the Evidence screen after the final run.
