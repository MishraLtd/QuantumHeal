# QuantumHealth Sentinel — MVP Prototype
### SIH26139 · Hybrid Quantum-Classical ML Platform for Early Disease Detection

A working, runnable proof-of-concept of the pipeline described in the idea deck:

```
Data ingestion → Pre-processing (scale + PCA) → Quantum feature encoding
→ Hybrid Variational Quantum Classifier → Explainability → Benchmarking
→ Clinician dashboard
```

## What's included

| File | Purpose |
|---|---|
| `hybrid_qml_mvp.py` | Core pipeline: trains a 4-qubit Variational Quantum Classifier (PennyLane) and two classical baselines (SVM, Random Forest) on the Breast Cancer Wisconsin dataset, benchmarks them, and generates explainability plots. |
| `dashboard.py` | Streamlit clinician-facing demo UI: patient screening, live benchmark table, explainability view. |
| `mvp_outputs/` | Generated on run: `benchmark_report.md`, ROC curves, confusion matrices, training curve, feature-importance chart. |
| `requirements.txt` | Exact pinned dependencies for a clean install. |

## Why this dataset for the MVP

The Breast Cancer Wisconsin (Diagnostic) dataset is a real, public, clinically-derived
dataset (569 patients, 30 digitized-biopsy biomarkers, malignant/benign labels) that
ships with scikit-learn — so the MVP runs **offline, in under a minute, with zero
downloads**, which matters for a live hackathon demo. The pipeline is dataset-agnostic:
swapping in TCGA genomics or MIMIC-III EHR data only means changing `load_data()`.

## Run it

```bash
pip install -r requirements.txt
python hybrid_qml_mvp.py        # trains + benchmarks, writes mvp_outputs/
streamlit run dashboard.py      # launches the clinician dashboard
```

## Sample result from a reference run

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Hybrid VQC (Quantum, 4 qubits) | 92.3% | 81.1% | 98.9% | 0.984 |
| Classical SVM (RBF) | 96.5% | 90.6% | 100% | 0.998 |
| Random Forest | 96.5% | 92.5% | 98.9% | 0.996 |

**Honest read for judges:** on this small, already-easy benchmark, the classical
baselines currently edge out the 4-qubit quantum model — expected, since classical
SVMs are near-optimal on this dataset and today's simulators/NISQ hardware limit
circuit depth and qubit count. The quantum model's AUC (0.984) is nonetheless
competitive, showing the entangling feature map is learning real structure. The
platform's value case is forward-looking: as qubit counts and coherence improve,
the *same* pipeline scales to higher-dimensional, more complex biomarker spaces
(e.g. full genomic panels) where classical kernels are expected to struggle more —
which is exactly the regime hybrid QML research targets.

## Scaling this MVP into the full platform

- Swap `load_data()` for TCGA / MIMIC-III / imaging feature loaders
- Increase `N_QUBITS` / `N_LAYERS` and move from `default.qubit` to `qiskit.ibmq`
  or AWS Braket devices for NISQ hardware runs
- Add SHAP on the classical baselines and Q-LIME on the quantum model for
  side-by-side explainability
- Wrap `vqc_predict` behind a REST endpoint for EHR/hospital-system integration
