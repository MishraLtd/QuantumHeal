# QuantumHealth Sentinel — MVP

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

## Multimodal upgrade — 3-phase plan

The single-tabular-dataset MVP above is being extended into a multimodal
platform (tabular + imaging, with more modalities and hardening to follow).
Work is broken into three phases so each one ships something demoable:

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Imaging modality added (X-ray-style), classical feature extraction (radiomics + HOG stand-in for a CNN embedding), independent imaging VQC arm, late-fusion combiner, missing-modality graceful degradation, multimodal dashboard tab | ✅ **Done** (this update) |
| **Phase 2** | Swap synthetic X-rays for a real dataset once reachable (NIH ChestX-ray14 / pneumonia set); swap HOG/radiomics for a pretrained CNN (ResNet/DenseNet) embedding; add blood-cell-morphology arm; SHAP (classical) + Q-LIME (quantum) explainability; attention-based fusion; uncertainty via shot-based confidence | ⬜ Planned |
| **Phase 3** | REST API for EHR integration; NISQ hardware backend (`qiskit.ibmq` / AWS Braket); federated-deployment architecture (weights-only sharing across hospitals); DPDP/HIPAA-style compliance documentation; synthetic-data augmentation via quantum-GAN | ⬜ Planned |

### Phase 1 — what was added

- `imaging_arm.py` — synthetic X-ray-style image generator (label-correlated
  opacity pattern) + `ImageFeaturePipeline` (radiomics-style intensity/edge/
  entropy stats + coarse HOG → PCA → quantum-angle-scaled features). Fully
  offline, no downloads, matching the original MVP's demo constraints — see
  the module docstring for exactly what to swap for real imaging data.
- `hybrid_qml_mvp.build_vqc_arm()` — generalized the original tabular VQC
  into a reusable factory so each modality gets its own small, independent
  quantum circuit.
- `fusion.py` — `LateFusion`: weighted-average combiner across whichever
  modalities are present for a given patient, with graceful degradation
  when one is missing (e.g. no imaging on file).
- `train_multimodal.py` — end-to-end training + benchmark script for the
  fused pipeline; writes `mvp_outputs/multimodal_benchmark_report.md` and
  `multimodal_roc_comparison.png`.
- `dashboard.py` — new **"🩻 Multimodal Screening"** tab: per-patient fused
  prediction, per-modality contribution breakdown, and an "imaging on file?"
  toggle that demonstrates missing-modality degradation live.

**Reference result** (synthetic imaging arm, offline run):

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Fused (Tabular + Imaging VQC) | 93.0% | 81.1% | 100% |
| Tabular VQC only | 92.3% | 81.1% | 98.9% |
| Imaging VQC only (synthetic) | 86.0% | 67.9% | 96.7% |

Fusion edges out either single arm — the architecture is doing its job —
though this is on synthetic imagery, so the honest read is "the plumbing
works," not "quantum imaging beats classical." Real-data validation is
Phase 2's job.

## Older scaling notes (superseded by the phase table above, kept for reference)

- Increase `N_QUBITS` / `N_LAYERS` and move from `default.qubit` to `qiskit.ibmq`
  or AWS Braket devices for NISQ hardware runs
- Wrap `vqc_predict` behind a REST endpoint for EHR/hospital-system integration
