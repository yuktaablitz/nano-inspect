# Answers to the questions we were asked

## 1. How is visual QC done in industry today, and how do we compare?

| Approach today | How it works | Weakness for our user | NanoInspect |
|---|---|---|---|
| **Human visual inspection** | Inspectors look at every part, or at a sample | In a Sandia study, 82 inspectors caught 85% of defects but also rejected 35% of good parts; the industry average hit rate is about 80% ([See, Sandia 2015](https://www.osti.gov/servlets/purl/1236203)) | Tier 1 checks every part the same way every time; people see only the cases the cost rule sends them |
| **Acceptance sampling (AQL, ISO 2859-1 / ANSI/ASQ Z1.4)** | Inspect a random sample per lot and accept the lot if defects stay under a limit. Example: a 4,000-unit lot at AQL 2.5 → inspect 200, accept if ≤ 10 defects ([QIMA](https://www.qima.com/aql-acceptable-quality-limit)) | Most parts are never inspected, and a passing lot can still ship defects | 100% inspection at line speed, plus a random audit of accepted parts |
| **Rule-based machine vision** (Cognex, Keyence style) | Hand-tuned thresholds and templates per product | Needs a vision engineer per product; brittle to new defect types and lighting | Learns a product from a few hundred good photos and ~40 defect photos; the LLM names the defect |
| **Cloud AI inspection services** | Images uploaded to a vendor cloud for inference | Images leave the plant, the line depends on the WAN and on one vendor. Amazon Lookout for Vision was discontinued on Oct 31, 2025 ([AWS](https://aws.amazon.com/blogs/machine-learning/exploring-alternatives-and-seamlessly-migrating-data-from-amazon-lookout-for-vision/)) | All inference on the Nano; only escalated ~10 KB crops leave, and only when the policy says so |
| **Open-source anomaly detection** ([Anomalib](https://github.com/open-edge-platform/anomalib)) | Anomaly score per image, edge-deployable | A score, not a decision: no defect type, no explanation, no SOP action, no human loop | Our training-free delta map is this idea, used as evidence; the LLMs add the what and the where, and the policy adds the decision |

Our measured numbers for the same comparison are in `artifacts/results/benchmark_summary.json` and on the app's *Models & serving* page. They cover the fine-tuned tier 1 vs zero-shot, the 27B tier 2, the ResNet and delta baselines, and cost per 1,000 parts against "a human checks every part".

**The story:** a company launching a new product does not want its product images, and so its designs, in a vendor's cloud yet. NanoInspect keeps inference, training and fine-tuning on one ZGX Nano in the plant. Nothing leaves unless a part is escalated, and then only a crop.

## 2. What is the SOP, where does it come from, and what is its role?

- **What:** the Standard Operating Procedure is the plant's approved instruction for what to do with a nonconforming part. It covers the disposition (reject, hold for the material review board, release), containment (for example, re-check the last N parts, or stop the line if a defect repeats), who to notify, and which process setting to check.
- **Where it comes from:** the plant's quality management system. ISO 9001:2015 clause 8.7, *Control of nonconforming outputs*, requires nonconforming outputs to be identified, segregated, recorded and dispositioned by authorised people ([summary](https://texasqa.com/iso-9001-clauses-explained-8-7-control-of-nonconforming-outputs/)). In NanoInspect this is `config/sop.json` (id `QA-SOP-NI-001`, version 1.0). It has one entry for each of the 73 defect types: severity class, likely causes, process check, and whether rework is allowed. For the demo we wrote it for the MVTec products; a real plant exports it from its QMS and version-controls it.
- **Role:** the models decide *what they see*; the SOP decides *what the plant does about it*. The LLM is never allowed to invent an action. It may only pick a cause the SOP lists, and every instruction carries the SOP id and version, so every decision is auditable against the approved procedure.

## 3. How is the "delta" between good and bad calculated?

Three measurements, from simplest to richest:
1. **Explicit delta map (training-free):** for each product we keep ~16,000 local feature vectors (patches) from 40 known-good parts, taken from an ImageNet-pretrained ResNet-18 (layers 2 and 3, no training). For a new part, every patch's distance to its nearest good patch is its delta. The heatmap shows *where* the part differs; the largest delta, expressed in standard deviations above held-out good parts, is the delta score. This is the PatchCore idea (Roth et al., CVPR 2022) in its simplest form: 23 ms per image, mean ROC-AUC 0.91 over 15 products using only good images. It drives the heatmap in the app and the crop sent to the cloud (`nanoinspect/delta.py`).
2. **Tier 1 P(defect):** the fine-tuned 7B model's probability of "defective", read from the log-probabilities of its verdict token. This is the number the fast-accept threshold uses. The threshold is set from validation good parts, never from test images.
3. **Tier 2 comparison:** the 27B model receives a known-good reference image and the part side by side, and is asked to compare them; it names the defect, its location, and explains the difference in a sentence.

## 4. The "3D printer" idea for our product: machine-readable output

When a part is rejected, NanoInspect produces a JSON instruction for the line (`nanoinspect.disposition/v1`, validated against a JSON schema before it is sent):
- `line_command`: `pass`, `divert_to_reject_bin`, `divert_to_hold_bin`, or `stop_line`, for the PLC / MES (in a plant this goes over OPC UA or MQTT).
- `nonconformance`: an NCR number, defect type, severity from the SOP, location, a description written by tier 2 from what it sees, a probable cause chosen from the SOP's list, the SOP process check, whether rework is allowed, and operator steps.
- `containment`: the SOP rule. If the same defect repeats N times within a window of parts, a `stop_line` command is issued automatically.

Example: `artifacts/results/example_machine_instruction.json`. Live: the *Production line* page shows every message the line controller received.

## 5. Synthetic defects: what we use them for, honestly

The first prototype painted rectangles and noise; we replaced those with the Perlin-texture (DRAEM) and CutPaste methods. These still look artificial, as the figure shows, and we only use them for the **ResNet baseline**. **The LLM that makes decisions (tier 1) is fine-tuned only on real defect photographs**: half of MVTec's real defects, the other half held out for testing. The defect transplant (a real defect pasted onto a good part) is the only synthetic method that looks realistic.
