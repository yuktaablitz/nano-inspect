| Metric                                              | Result                                                                |
|:----------------------------------------------------|:----------------------------------------------------------------------|
| Products / defect types                             | 15 products, 73 real defect types (MVTec AD), held-out evaluation     |
| Tier 1 ROC-AUC · recall (fine-tuned 7B)             | 0.962 · 84.4%   (zero-shot 7B: 0.837 · 20.8%)                         |
| Tier 1 names the right defect type · location       | 73.6% · 70.6%                                                         |
| Tier 2 ROC-AUC · recall (27B + reference)           | 0.988 · 94.3%   (27B zero-shot: 0.924 · 74.7%)                        |
| Classical baseline ResNet-18 ROC-AUC · recall       | 0.960 · 92.8% (no defect type or location)                            |
| LoRA fine-tune on the Nano                          | 28 min, 0.484% of parameters                                          |
| Serving throughput (vLLM on the Nano)               | tier 1 8.36 img/s at 32 clients · tier 2 0.46 img/s at 8 clients      |
| Cost per 1,000 parts: cascade vs human inspects all | $411.91 vs $500.00                                                    |
| Per 1,000 parts (cascade)                           | 7.23 escapes · 11.35 scrapped · 55.8 human reviews · 151 tier-2 calls |
| Soak 15 min, both tiers                             | 2,671 + 180 inferences · max 70 C · throttling: False · errors: 0     |
| Energy per inspection (GPU)                         | tier 1 13.7 J · tier 2 122.2 J                                        |
| Offline (network blocked)                           | 3/3 full inspections, 0 outbound connection attempts                  |
| Failure cases                                       | 9 tested, 0 broken inputs accepted                                    |