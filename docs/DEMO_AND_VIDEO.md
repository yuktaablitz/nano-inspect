# Demo runbook, 2-minute video script and animation prompt

## Before the demo (on the Nano)
1. Start everything and check it end to end:
   ```bash
   ./start_all.sh --https     # both model tiers, cloud review tier, operator app (HTTPS, reachable from a laptop or phone)
   ./start_all.sh demo        # runs the accept / reject / human-review parts and prints PASS for each
   ```
   A cold start takes about 10 minutes (model loading), and restarting only the app takes about 5. Start well before the demo, and afterwards reload the browser rather than restarting.
2. Open **https://localhost:8080** on the Nano, or **https://&lt;nano-ip&gt;:8080** from a laptop or phone. Accept the self-signed certificate once.
3. Open the cloud review console in a second tab: **Operations → Cloud review console ↗**.
4. Keep **Operations → Production line** stopped unless you are demonstrating it: the simulator shares tier 2 with the chat and slows it down.
5. Run each demo part once before recording, so that repeat runs are warm.

Typical times on the Nano (tier 2 = fine-tuned 27B in FP8):

| Part | Outcome | Time to decision |
|---|---|---|
| Good part | ACCEPT at tier 1 | ~2 s |
| Clear defect | REJECT at tier 2 + machine instruction | ~6–10 s; the full explanation fills in a few seconds later |
| Models disagree | HUMAN REVIEW (crop sent to the cloud) | ~6–10 s |

## 5-minute live demo
| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Home | Scroll to the three tiers | The user, the one-second budget, and three tiers where the cloud comes last |
| 0:40 | Home → **Watch 3 decisions** | Run accept, reject, review | A good part is accepted by the small model; a broken rim is rejected with an SOP instruction; a disagreement goes to a person |
| 1:40 | **Inspect** | Upload a photo → **Submit for inspection**, then ask the chat "Could this be lighting?" | The image never leaves the Nano; the 27B answers in words within a second |
| 2:30 | **Quality rules → Escalation policy & costs** | Raise the escape cost from 50 to 500, Apply | The rule changes which evidence justifies a person; cost per 1,000 parts vs baselines |
| 3:10 | **Quality rules → SOP & line instructions** | Download the SOP; upload it again | The plant owns the procedure; an uploaded SOP is validated and applies to the next decision |
| 3:40 | Top bar | Switch **Network online** off, inspect a part, then **Operations → Human review queue** | Inspection continues; escalations queue on the Nano; back online, only a small crop is sent |
| 4:20 | **Performance → Accuracy & fine-tuning**, then **Stress tests** | Scroll | Fine-tuning on the Nano, results vs baselines, 15-minute soak, offline proof |

**Backup:** record a run beforehand in case the room network fails.

## 2-minute video script
| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Home page hero | "On a production line, a quality inspector has about one second per part. People miss up to one defect in five, and sampling checks only a fraction of parts. Cloud AI could help, but for a company launching a new product, every photo *is* the design. Uploading it is a leak, and when the internet drops, the line can't stop." |
| 0:15–0:30 | Home: the three tiers | "So we put the inspector at the edge: one HP ZGX Nano next to the line. Two vision-language models, fine-tuned on the Nano itself, check every part locally. A fast 7-billion-parameter model sees everything; a 27-billion-parameter model looks only when it's unsure. No image leaves the building." |
| 0:30–1:00 | **Inspect**: upload → Submit → result → chat | "I upload a photo from the line and press Submit. A good part is accepted in about two seconds by the small model. A cracked bottle goes to the big model, which compares it with a known-good part, names the defect and where it is, and sends the line a machine instruction to divert it. The operator can ask it why, right here." |
| 1:00–1:20 | **Quality rules**: Escalation policy, then SOP | "When does a person look? Your costs decide. A part goes to a person only when a review is cheaper than the risk of a wrong call, and only a small crop is sent. The procedure is yours too: download the SOP, or upload your plant's own." |
| 1:20–1:40 | **Operations → Live overview**, then **Performance → Accuracy & fine-tuning** | "Operations shows the line live: decisions per tier, device health, and the review queue. Over 99% of image data stays on site. And the results: 94% of defects caught with under 5% false alarms, up to 54% cheaper than checking every part by hand." |
| 1:40–2:00 | Switch **Network online** off, inspect a part | "Because it runs at the edge, it keeps working when the network doesn't: inspection carries on, and escalations wait until the link is back. One box per line, and your data stays yours. Every part inspected on the line; only the hard cases leave the building." |

## Prompt for an animated product video (text-to-video generator)
> A 30-second clean, modern 3D animation in a black-and-orange tech style. A factory production line with parts (bottles, screws, pills) moving on a conveyor. A small glowing box labeled "HP ZGX Nano" sits beside the line, and a camera above scans each part with an orange light beam. Most parts get a green check and roll on. A cracked bottle gets a red box around the crack and is pushed into a reject bin. One unclear part sends only a tiny glowing image tile up through a gateway to a cloud, where a person reviews it and sends a green label back down. A dashed boundary around the factory shows full images never leave the building. The internet icon switches off, and the line keeps running. End on the text: "NanoInspect: every part inspected on the line. Only the hard cases leave the building." Smooth camera moves, minimal, professional.

Add the closing tagline in editing: generators often render on-screen text poorly.
