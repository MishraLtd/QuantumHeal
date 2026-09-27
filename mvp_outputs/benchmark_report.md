# Benchmark Report — Hybrid QML vs. Classical Baselines

Dataset: Breast Cancer Wisconsin (Diagnostic), public sklearn dataset (569 samples, 30 biomarker features, malignant vs. benign).

| Model | Accuracy | Sensitivity (Recall) | Specificity | Precision | Train time (s) |
|---|---|---|---|---|---|
| Hybrid VQC (Quantum) | 0.923 | 0.811 | 0.989 | 0.977 | 11.27 |
| Classical SVM (RBF) | 0.965 | 0.906 | 1.000 | 1.000 | 0.13 |
| Random Forest | 0.965 | 0.925 | 0.989 | 0.980 | 0.13 |
