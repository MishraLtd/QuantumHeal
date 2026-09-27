# Multimodal Benchmark Report — Phase 1

Tabular data: Breast Cancer Wisconsin (real, public). Imaging data: **synthetic** X-ray-style images (see `imaging_arm.py` for why — no network access to real medical imaging repositories from this environment). Numbers below demonstrate the *architecture* working end-to-end, not a claim of real diagnostic imaging performance.

| Model | Accuracy | Sensitivity | Specificity | Precision |
|---|---|---|---|---|
| Fused (Tabular + Imaging VQC) | 0.930 | 0.811 | 1.000 | 1.000 |
| Tabular VQC only | 0.923 | 0.811 | 0.989 | 0.977 |
| Imaging VQC only | 0.860 | 0.679 | 0.967 | 0.923 |
| Classical SVM (RBF) (tabular, classical) | 0.965 | 0.906 | 1.000 | 1.000 |
| Random Forest (tabular, classical) | 0.965 | 0.925 | 0.989 | 0.980 |

## Missing-modality robustness
- 47/143 test patients simulated with imaging unavailable -> fused using tabular arm only for those cases.
- Fused accuracy with that partial-data cohort: **0.930** (platform degrades gracefully instead of failing when imaging is absent).

## Fusion contribution weights (both modalities present)
- {'tabular': 0.6, 'imaging': 0.4}
