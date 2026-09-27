# QuantumHealth Sentinel — Hybrid Quantum-Classical Multimodal Diagnostic Platform

Multimodal, quantum-enhanced early disease detection fusing structured lab data, radiology imaging, and hematology imaging into a single hybrid classifier — benchmarked live against classical baselines, with clinician-facing explainability and confidence-calibrated predictions.

## Overview

QuantumHealth Sentinel screens patients by combining three independent diagnostic signals — blood test panels, X-ray imaging, and blood cell microscopy — into a fused representation that is encoded onto qubits and classified via a Variational Quantum Classifier (VQC), running alongside classical SVM and Random Forest baselines for transparent, real-time comparison.

The platform is disease-agnostic by design: the same ingestion → fusion → quantum-encoding → classification → explainability pipeline is built to be re-targeted across conditions (oncology, pneumonia, anemia, and beyond) by swapping data adapters, not by rebuilding the system.

## Core Features

### 1. Multimodal Diagnostic Fusion
- **Structured Arm** — blood test reports (CBC, metabolic panels, lipid panels) parsed, scaled, and dimensionality-reduced via PCA.
- **Radiology Arm** — X-ray imaging processed through a dedicated CNN feature extractor into compact diagnostic embeddings.
- **Hematology Arm** — blood cell microscopy processed through a morphology-tuned CNN extractor, kept separate from the radiology model given the two domains' differing visual statistics.
- **Graceful Degradation** — patients screened on partial data (e.g., labs only, no imaging) are still processed, with the dashboard flagging which modalities informed the result.

### 2. Fusion Engine
- **Late Fusion** — per-modality quantum classifiers combined via a lightweight classical aggregation head, with per-modality contribution fully traceable.
- **Attention Fusion** — a learned attention layer dynamically weights each modality per patient (e.g., prioritizing an abnormal blood panel over a normal X-ray), the platform's core differentiator over static multimodal pipelines.

### 3. Hybrid Quantum-Classical Classification
- Fused, PCA-reduced feature vectors encoded onto qubits via angle/amplitude encoding.
- Variational Quantum Classifier trained with a classical optimizer, simulator-executable with a direct path to NISQ hardware (Qiskit IBMQ, AWS Braket).
- Classical SVM and Random Forest baselines run in parallel for continuous head-to-head validation.

### 4. Live Benchmarking & Evaluation Suite
- Accuracy, sensitivity, specificity, AUC, ROC curves, confusion matrices, and training curves reported across every model, quantum and classical, side by side.

### 5. Per-Modality Explainability
- SHAP-based attribution for classical components, permutation/Q-LIME-style analysis for the quantum classifier.
- Dashboard reports not just *what* was predicted but *which modality and which features* drove the result.

### 6. Confidence-Calibrated Predictions
- Shot-based sampling from the variational quantum circuit produces a genuine confidence distribution per prediction, not a bare label — giving clinicians a calibrated sense of how certain the system is.

### 7. Clinician Dashboard
- Unified patient screening across all three modalities, live benchmark comparison, per-modality explainability views, and confidence-flagged results, built around real clinical review workflows.

## Tech Stack

**Quantum & ML Core:** PennyLane (VQC, quantum feature encoding), scikit-learn (SVM, Random Forest baselines), PyTorch (CNN feature extractors for radiology & hematology arms)

**Explainability:** SHAP, permutation importance, Q-LIME-style quantum attribution

**Dashboard & Visualization:** Streamlit, Plotly / Matplotlib for ROC, confusion matrix, and training-curve visualization

**Data Layer:** Pandas/NumPy preprocessing pipeline, modular `load_data()` adapters per modality and per disease domain

**Hardware Backends:** `default.qubit` simulator (current), Qiskit IBMQ and AWS Braket (NISQ deployment path)

## Getting Started

### Prerequisites
- Python 3.10+
- pip or conda
- (Optional) Access credentials for a NISQ backend (Qiskit IBMQ / AWS Braket) if running on real quantum hardware

### Installation

```bash
# Clone the repository
git clone https://github.com/MishraLtd/QuantumHeal.git
cd QuantumHeal

# Install dependencies
pip install -r requirements.txt
```

### Running the Platform

```bash
# Run the full pipeline: ingestion, fusion, quantum encoding, training, benchmarking
python hybrid_qml_mvp.py

# Launch the clinician-facing dashboard
streamlit run dashboard.py
```

The dashboard will start on `http://localhost:8501`.

## Project Structure

```
├── data/                     # Modality adapters — structured, radiology, hematology loaders
├── models/
│   ├── quantum/               # VQC, quantum feature encoding, circuit definitions
│   ├── classical/              # SVM, Random Forest baseline models
│   └── extractors/             # CNN feature extractors (radiology, hematology arms)
├── fusion/                    # Late-fusion and attention-fusion modules
├── explainability/             # SHAP, permutation, and quantum attribution modules
├── mvp_outputs/                 # Generated benchmark reports, ROC curves, confusion matrices
├── dashboard.py                # Streamlit clinician dashboard
├── hybrid_qml_mvp.py            # Core pipeline entrypoint
└── requirements.txt              # Pinned dependencies
```

## Future Integrations & Roadmap

- **Large-Scale Medical Dataset Training** — expanding beyond initial benchmark datasets to large, diverse, clinically-sourced datasets (genomic panels, hospital EHR archives, multi-institution imaging repositories) to substantially improve model precision, sensitivity, and generalization across patient populations.
- **NISQ Hardware Deployment** — moving benchmark and inference runs from simulators onto real quantum hardware (Qiskit IBMQ, AWS Braket) as qubit counts and coherence times improve.
- **Multi-Disease Expansion** — extending the disease-agnostic pipeline across additional conditions (pneumonia, anemia, cardiac risk, and beyond) via new data adapters.
- **REST API / EHR Integration** — exposing the trained classifier as a service endpoint for direct integration into hospital information systems.
- **Federated Deployment Architecture** — enabling multi-hospital deployment where only model weights, never patient data, are shared across institutions.
- **Compliance & Data Governance** — formal alignment with the DPDP Act (India) and equivalent international standards (e.g., HIPAA) for clinical-grade deployment.
- **Synthetic Data Augmentation** — quantum-circuit-based generation of synthetic training samples to address the scarcity of large labelled medical imaging datasets.
- **Continuous Validation** — an ongoing, expanding multi-domain validation report as the platform is trained on progressively larger and more diverse medical datasets.

## License
