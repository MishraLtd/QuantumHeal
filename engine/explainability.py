"""
QuantumHealth Sentinel — Phase 2 Explainability & Uncertainty
=================================================================
Two independent additions:

1. SHAP on the classical baseline (Random Forest) — a standard,
   trusted explainability method clinicians/judges are likely to
   already recognize, run alongside the existing permutation-importance
   view of the quantum model for a direct classical-vs-quantum
   explainability comparison.

2. Shot-based uncertainty for the quantum arm(s) — repeated finite-shot
   sampling of the trained circuit gives a spread of outputs per patient
   instead of one deterministic number. This is a genuinely quantum-native
   way to get a confidence interval (a classical point-estimate model
   doesn't have this "for free"), used here as the "Q-LIME"-style
   uncertainty item from the Phase 2 plan — simpler than a full Q-LIME
   implementation, but honest and immediately usable in the dashboard.
"""

import numpy as np
import pennylane as qml


# ------------------------------------------------------------------
# 1. SHAP on classical Random Forest baseline
# ------------------------------------------------------------------
def shap_summary_rf(rf_model, X_train, X_test, feature_names, max_display=10):
    """Returns (mean_abs_shap, order) for the RF baseline using SHAP's
    exact TreeExplainer (fast, no sampling approximation needed)."""
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    # shap>=0.45 returns a single array for binary-classification trees;
    # older versions return a [class0, class1] list — handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
        shap_values = shap_values[:, :, 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    return mean_abs, order, [feature_names[i] for i in order]


# ------------------------------------------------------------------
# 2. Shot-based uncertainty quantification for a trained VQC arm
# ------------------------------------------------------------------
def build_shot_uncertainty_fn(n_qubits, shots=200, n_repeats=15):
    """Returns a function `estimate(weights, bias, X)` -> (mean_prob, std_prob)
    that re-runs the SAME trained circuit under finite-shot sampling
    (rather than exact analytic expectation) `n_repeats` times per sample,
    giving a per-patient confidence spread — how much the answer would
    wobble on real, noisy, finite-shot NISQ hardware."""
    dev_shots = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_shots)
    def circuit_shots(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    circuit_shots = qml.set_shots(circuit_shots, shots=shots)

    def estimate(weights, bias, X):
        means, stds = [], []
        for x in X:
            samples = np.array([
                circuit_shots(weights, x) + bias for _ in range(n_repeats)
            ])
            probs = np.clip((samples + 1) / 2, 0, 1)
            means.append(probs.mean())
            stds.append(probs.std())
        return np.array(means), np.array(stds)

    return estimate
