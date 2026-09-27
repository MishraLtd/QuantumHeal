# Phase 2 Benchmark Report

Imaging data is still **synthetic** (see `imaging_arm.py` — real medical
imaging hosts and pretrained-CNN-weight hosts are both unreachable from
this sandboxed network; confirmed by a direct 403 from zenodo.org).
This report's purpose is to validate the Phase 2 additions
(CNN embeddings, SHAP, shot-based uncertainty) work end-to-end so the
swap to real data is a data-loader change only, not an architecture change.

## Embedding comparison — does a learned CNN beat handcrafted features?

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Fused (Tabular + CNN-Imaging VQC) | 0.958 | 0.887 | 1.000 |
| Tabular VQC only | 0.923 | 0.811 | 0.989 |
| Imaging VQC (HOG/radiomics, Phase 1) | 0.860 | 0.679 | 0.967 |
| Imaging VQC (CNN embedding, Phase 2) | 0.972 | 0.925 | 1.000 |
| Classical SVM (RBF) (tabular, classical) | 0.965 | 0.906 | 1.000 |
| Random Forest (tabular, classical) | 0.965 | 0.925 | 0.989 |

**Honest read:** the CNN-imaging arm (and therefore the fusion) looks
very strong here — stronger than the real-data tabular arm and even the
classical baselines. Treat that with suspicion, not celebration: the CNN
is learning a synthetic image-generation rule that this same codebase
wrote (see `imaging_arm.py`), so near-perfect performance mostly proves
the CNN can learn *that* rule, not that imaging beats biomarkers
diagnostically. The result that's actually meaningful for the platform
story is architectural: a learned CNN embedding materially outperformed
handcrafted HOG/radiomics features on the *same* synthetic task (97.2%
vs. 86.0% accuracy) — the multimodal pipeline correctly benefits from a
better feature extractor, which is the property that should carry over
once real imaging data replaces the synthetic generator in Phase 2's
next slice.

## Explainability — SHAP (classical baseline)
- Top biomarkers by mean |SHAP value|: mean radius, mean texture, mean perimeter, mean area
- Chart: `shap_rf_importance.png` — compare directly against the existing
  quantum-model permutation-importance chart (`feature_importance.png`)
  from Phase 1's `hybrid_qml_mvp.py` run for a classical-vs-quantum
  explainability comparison in the dashboard/pitch.

## Uncertainty quantification — shot-based confidence (tabular VQC arm)
- Mean predicted-probability std across 15 resamples of 200 shots each: **0.027**
- ~10% of test patients fall in the top decile of uncertainty (std > 0.038) — these are the cases a clinician
  dashboard should flag as "low-confidence, recommend a second opinion /   additional test" rather than presenting a bare label.
- Chart: `uncertainty_scatter.png`.
