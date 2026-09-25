| Metric                                         | Result                                                                    |
|:-----------------------------------------------|:--------------------------------------------------------------------------|
| Real-defect ROC-AUC (83 MVTec test images)     | 0.993                                                                     |
| Recall / precision at 0.5                      | 0.968 / 0.984                                                             |
| Escapes at decision policy                     | 2.0 of 63 defects                                                         |
| Manual-review share                            | 10.8%                                                                     |
| Vision latency, batch 1, from disk (p50 / p95) | 17.5 / 19.8 ms                                                            |
| Peak GPU throughput                            | 13,076 img/s (bs=64, bf16)                                                |
| Soak test (15.0 min)                           | 3,674,432 images, throughput change -0.1%, max 68 C, throttle flag: False |
| Worst robustness case                          | low contrast (AUC 0.938)                                                  |
| Agent: valid reports                           | 100% of 24                                                                |
| Agent: LLM text valid / fallback               | 100% / 0%                                                                 |
| Agent: end-to-end p50 / p95                    | 9.3 / 10.0 s                                                              |
| Offline run (network blocked)                  | 3/3 completed, 0 connection attempts                                      |
| Failure cases                                  | 8 tested, silent accepts: False                                           |