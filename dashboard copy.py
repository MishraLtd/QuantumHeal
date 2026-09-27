"""
QuantumHealth Sentinel — Clinician Dashboard (MVP demo)
========================================================
Streamlit front-end for the hybrid quantum-classical MVP.
Run with:  streamlit run dashboard.py
(Requires hybrid_qml_mvp.py to have been run once so mvp_outputs/ exists,
 or click "Train now" in the sidebar to train live inside the app.)
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import hybrid_qml_mvp as core

st.set_page_config(page_title="QuantumHealth Sentinel", layout="wide")

st.title("🧬 QuantumHealth Sentinel")
st.caption("Hybrid Quantum-Classical ML Platform for Early Disease Detection — SIH26139 MVP")

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        "1. **Ingest** biomedical data\n"
        "2. **Pre-process**: scale + PCA reduce\n"
        "3. **Quantum encode**: angle embedding\n"
        "4. **Hybrid VQC**: variational quantum classifier\n"
        "5. **Explain**: which biomarkers drove the call\n"
        "6. **Benchmark**: vs. classical SVM / Random Forest"
    )
    run_button = st.button("🔄 Train / Refresh model (~20s)")

@st.cache_resource(show_spinner="Training hybrid quantum-classical model...")
def get_trained_model():
    X, y, feature_names = core.load_data()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=core.SEED, stratify=y
    )
    X_train, X_test, scaler, pca = core.preprocess(X_train_raw, X_test_raw)
    weights, bias, loss_history = core.train_vqc(X_train, y_train)
    calibrator = core.calibrate_vqc(weights, bias, X_train, y_train)
    baselines = core.train_classical_baselines(X_train, y_train)
    return {
        "X_test_raw": X_test_raw, "X_test": X_test, "y_test": y_test,
        "weights": weights, "bias": bias, "loss_history": loss_history,
        "calibrator": calibrator,
        "baselines": baselines, "scaler": scaler, "pca": pca,
        "feature_names": feature_names,
    }

if run_button:
    get_trained_model.clear()

state = get_trained_model()

tab1, tab2, tab3 = st.tabs(["📋 Patient Screening", "📊 Benchmark", "🔍 Explainability"])

with tab1:
    st.subheader("Screen a patient sample")
    idx = st.slider("Pick a test-set sample to screen", 0, len(state["y_test"]) - 1, 0)
    x = state["X_test"][idx:idx+1]
    preds, probs = core.vqc_predict(state["weights"], state["bias"], x, calibrator=state["calibrator"])
    true_label = state["y_test"][idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Quantum model prediction", "⚠️ Disease" if preds[0] == 1 else "✅ No disease")
    col2.metric("Confidence", f"{probs[0]*100:.1f}%")
    col3.metric("Ground truth (held-out label)", "Disease" if true_label == 1 else "No disease")

    st.markdown("**Raw biomarker snapshot (first 8 of 30 features):**")
    raw_row = pd.DataFrame(
        [state["X_test_raw"][idx][:8]],
        columns=state["feature_names"][:8]
    )
    st.dataframe(raw_row, use_container_width=True)

with tab2:
    st.subheader("Hybrid Quantum vs. Classical — head to head")
    rows = []
    vqc_preds, vqc_probs = core.vqc_predict(state["weights"], state["bias"], state["X_test"], calibrator=state["calibrator"])
    from sklearn.metrics import accuracy_score, precision_score
    sens, spec = core.sensitivity_specificity(state["y_test"], vqc_preds)
    rows.append(["Hybrid VQC (Quantum)",
                 accuracy_score(state["y_test"], vqc_preds), sens, spec])
    for name, model in state["baselines"].items():
        preds = model.predict(state["X_test"])
        sens, spec = core.sensitivity_specificity(state["y_test"], preds)
        rows.append([name, accuracy_score(state["y_test"], preds), sens, spec])
    df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Sensitivity", "Specificity"])
    st.dataframe(df.style.format({"Accuracy": "{:.1%}", "Sensitivity": "{:.1%}", "Specificity": "{:.1%}"}),
                 use_container_width=True)
    st.line_chart(pd.DataFrame({"VQC training loss": state["loss_history"]}))

with tab3:
    st.subheader("Why did the model say that?")
    st.write(
        "Principal components fed to the quantum circuit are traced back to "
        "the original clinical biomarkers so a clinician can see *what drove* "
        "the prediction, not just the output."
    )
    top_pc = 0
    loadings = state["pca"].components_[top_pc]
    order = np.argsort(np.abs(loadings))[::-1][:8]
    df2 = pd.DataFrame({
        "Biomarker": [state["feature_names"][i] for i in order],
        "Influence on PC1": [loadings[i] for i in order],
    })
    st.bar_chart(df2.set_index("Biomarker"))

st.divider()
st.caption("MVP prototype — Qiskit/PennyLane-based hybrid QML, "
           "runs on classical simulators today; portable to IBM Quantum / AWS Braket NISQ hardware.")
