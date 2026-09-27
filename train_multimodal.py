"""
QuantumHealth Sentinel — Multimodal Training & Benchmark  (Phase 1)
=====================================================================
Runs the full multimodal pipeline end-to-end:

    Tabular arm:  Breast Cancer Wisconsin biomarkers -> PCA -> VQC (4 qubits)
    Imaging arm:  synthetic X-ray-style images        -> HOG+PCA -> VQC (4 qubits)
    Fusion:       late fusion of the two arms' probabilities
    Benchmark:    fused model vs. tabular-only vs. imaging-only vs.
                  classical baselines (SVM, Random Forest)
    Robustness:   demonstrates missing-modality graceful degradation

Run:
    python train_multimodal.py

Outputs (written to ./mvp_outputs/):
    - multimodal_benchmark_report.md
    - multimodal_roc_comparison.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc

import hybrid_qml_mvp as core
from imaging_arm import load_synthetic_xrays, ImageFeaturePipeline, N_QUBITS_IMG
from fusion import LateFusion

OUT_DIR = "mvp_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("QuantumHealth Sentinel — Multimodal MVP (Phase 1)")
    print("=" * 70)

    # ---- 1. Load + split tabular data (shared patient split across arms) ----
    print("\n[1/6] Loading tabular biomarker data...")
    X_tab, y, feature_names = core.load_data()
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.25, random_state=core.SEED, stratify=y
    )
    X_tab_train_raw, X_tab_test_raw = X_tab[idx_train], X_tab[idx_test]
    print(f"  {len(y_train)} train / {len(y_test)} test patients")

    # ---- 2. Generate the synthetic imaging modality for the SAME patients ----
    print("\n[2/6] Generating synthetic X-ray-style imagery "
          "(paired 1:1 with the tabular patients; see imaging_arm.py docstring)...")
    images_all = load_synthetic_xrays(y, seed=core.SEED)
    img_train, img_test = images_all[idx_train], images_all[idx_test]

    # ---- 3. Pre-process + train the TABULAR VQC arm ----
    print("\n[3/6] Training tabular VQC arm...")
    X_tab_train, X_tab_test, tab_scaler, tab_pca = core.preprocess(X_tab_train_raw, X_tab_test_raw)
    tab_arm = core.build_vqc_arm(n_qubits=core.N_QUBITS)
    t0 = time.time()
    tab_weights, tab_bias, tab_loss = tab_arm["train"](X_tab_train, y_train)
    tab_train_time = time.time() - t0
    tab_preds, tab_probs = tab_arm["predict"](tab_weights, tab_bias, X_tab_test)

    # ---- 4. Pre-process + train the IMAGING VQC arm ----
    print("\n[4/6] Training imaging VQC arm (HOG features -> PCA -> quantum encoding)...")
    img_pipeline = ImageFeaturePipeline(n_components=N_QUBITS_IMG)
    X_img_train = img_pipeline.fit_transform(img_train)
    X_img_test = img_pipeline.transform(img_test)
    img_arm = core.build_vqc_arm(n_qubits=N_QUBITS_IMG)
    t0 = time.time()
    img_weights, img_bias, img_loss = img_arm["train"](X_img_train, y_train)
    img_train_time = time.time() - t0
    img_preds, img_probs = img_arm["predict"](img_weights, img_bias, X_img_test)

    # ---- 5. Late fusion (+ missing-modality robustness demo) ----
    print("\n[5/6] Fusing modalities + testing missing-modality degradation...")
    fusion = LateFusion()
    fused_probs, fused_preds, contributions = fusion.fuse(tab_probs, img_probs)
    print(f"  Fused, both modalities present -> contributions: {contributions}")

    # Simulate ~30% of the test cohort arriving with NO imaging on file
    # (routine real-world scenario) and confirm the platform still predicts.
    rng = np.random.RandomState(core.SEED)
    missing_mask = rng.rand(len(y_test)) < 0.30
    degraded_probs = fused_probs.copy()
    degraded_contrib_note = (
        f"{missing_mask.sum()}/{len(y_test)} test patients simulated with "
        f"imaging unavailable -> fused using tabular arm only for those cases."
    )
    if missing_mask.any():
        tab_only_probs, tab_only_preds, _ = fusion.fuse(tabular_prob=tab_probs[missing_mask])
        degraded_probs[missing_mask] = tab_only_probs
    degraded_preds = (degraded_probs > 0.5).astype(int)
    degraded_acc = accuracy_score(y_test, degraded_preds)
    print(f"  {degraded_contrib_note}")
    print(f"  Accuracy with simulated missing imaging: {degraded_acc:.3f}")

    # ---- 6. Classical baselines + full benchmark report ----
    print("\n[6/6] Training classical baselines + writing benchmark report...")
    baselines = core.train_classical_baselines(X_tab_train, y_train)

    results = {}
    results["Fused (Tabular + Imaging VQC)"] = {"preds": fused_preds, "probs": fused_probs}
    results["Tabular VQC only"] = {"preds": tab_preds, "probs": tab_probs}
    results["Imaging VQC only"] = {"preds": img_preds, "probs": img_probs}
    for name, model in baselines.items():
        preds = model.predict(X_tab_test)
        probs = model.predict_proba(X_tab_test)[:, 1]
        results[f"{name} (tabular, classical)"] = {"preds": preds, "probs": probs}

    lines = [
        "# Multimodal Benchmark Report — Phase 1",
        "",
        "Tabular data: Breast Cancer Wisconsin (real, public). Imaging data: "
        "**synthetic** X-ray-style images (see `imaging_arm.py` for why — no "
        "network access to real medical imaging repositories from this "
        "environment). Numbers below demonstrate the *architecture* working "
        "end-to-end, not a claim of real diagnostic imaging performance.",
        "",
        "| Model | Accuracy | Sensitivity | Specificity | Precision |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        acc = accuracy_score(y_test, r["preds"])
        sens, spec = core.sensitivity_specificity(y_test, r["preds"])
        prec = precision_score(y_test, r["preds"], zero_division=0)
        lines.append(f"| {name} | {acc:.3f} | {sens:.3f} | {spec:.3f} | {prec:.3f} |")

    lines += [
        "",
        "## Missing-modality robustness",
        f"- {degraded_contrib_note}",
        f"- Fused accuracy with that partial-data cohort: **{degraded_acc:.3f}** "
        f"(platform degrades gracefully instead of failing when imaging is absent).",
        "",
        "## Fusion contribution weights (both modalities present)",
        f"- {contributions}",
    ]
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(os.path.join(OUT_DIR, "multimodal_benchmark_report.md"), "w") as f:
        f.write(report_text + "\n")

    # ---- ROC comparison plot ----
    plt.figure(figsize=(7, 5.5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Multimodal Fusion vs. Single-Arm vs. Classical")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "multimodal_roc_comparison.png"), dpi=150)
    plt.close()

    print(f"\nAll multimodal outputs written to ./{OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
