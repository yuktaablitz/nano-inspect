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