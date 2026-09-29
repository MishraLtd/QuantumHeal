"""
QuantumHeal — One-Time Model Export

Trains the current QuantumHeal model once and saves all artifacts
required for inference so the cloud API does NOT retrain at startup.
"""


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import pickle
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from engine import hybrid_qml_mvp as core
from engine.imaging_arm import load_synthetic_xrays, N_QUBITS_IMG
from engine.cnn_embedding import CNNFeaturePipeline
from engine.fusion import LateFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("QuantumHeal — One-Time Model Export")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("\n[1/5] Loading dataset...")

    X, y, feature_names = core.load_data()

    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.25,
        random_state=core.SEED,
        stratify=y,
    )

    X_train_raw = X[idx_train]
    X_test_raw = X[idx_test]

    # ---------------------------------------------------------
    # 2. TABULAR ARM
    # ---------------------------------------------------------
    print("\n[2/5] Training tabular VQC arm...")

    X_train, X_test, scaler, pca = core.preprocess(
        X_train_raw,
        X_test_raw
    )

    # Same scaling logic used by app/api.py
    pca_raw_train = pca.transform(
        scaler.transform(X_train_raw)
    )

    tab_scale = np.max(
        np.abs(pca_raw_train)
    ) + 1e-9

    tab_arm = core.build_vqc_arm(
        n_qubits=core.N_QUBITS
    )

    tab_weights, tab_bias, _ = tab_arm["train"](
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 3. IMAGING + CNN
    # ---------------------------------------------------------
    print("\n[3/5] Training imaging CNN...")

    images_all = load_synthetic_xrays(
        y,
        seed=core.SEED
    )

    img_train = images_all[idx_train]

    cnn = CNNFeaturePipeline(
        embed_dim=N_QUBITS_IMG
    )

    cnn_features = cnn.fit_transform(
        img_train,
        y_train
    )

    # ---------------------------------------------------------
    # 4. IMAGING VQC
    # ---------------------------------------------------------
    print("\n[4/5] Training imaging VQC arm...")

    img_arm = core.build_vqc_arm(
        n_qubits=N_QUBITS_IMG
    )

    img_weights, img_bias, _ = img_arm["train"](
        cnn_features,
        y_train
    )

    # ---------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # ---------------------------------------------------------
    print("\n[5/5] Saving model artifacts...")

    state = {
        "feature_names": list(feature_names),

        # Tabular preprocessing
        "scaler": scaler,
        "pca": pca,
        "tab_scale": float(tab_scale),

        # Tabular VQC
        "tab_weights": np.asarray(tab_weights),
        "tab_bias": float(tab_bias),

        # Imaging VQC
        "img_weights": np.asarray(img_weights),
        "img_bias": float(img_bias),

        # Fusion
        "fusion_weights": {
            "tabular": 0.6,
            "imaging": 0.4,
        },

        # Metadata
        "seed": core.SEED,
        "n_qubits_tabular": core.N_QUBITS,
        "n_qubits_imaging": N_QUBITS_IMG,
    }

    with open(
        ARTIFACT_DIR / "qhs_model.pkl",
        "wb"
    ) as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # Save CNN separately
    torch.save(
        cnn.model.state_dict(),
        ARTIFACT_DIR / "qhs_cnn.pt"
    )

    metadata = {
        "model_version": "0.3.0-cloud-1",
        "tabular_features": len(feature_names),
        "tabular_qubits": core.N_QUBITS,
        "imaging_qubits": N_QUBITS_IMG,
        "fusion": "LateFusion",
        "tabular_weight": 0.6,
        "imaging_weight": 0.4,
        "imaging_dataset": "synthetic",
        "training_seed": core.SEED,
    }

    with open(
        ARTIFACT_DIR / "metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\nArtifacts created:")

    for path in ARTIFACT_DIR.iterdir():
        print(f"  ✓ {path.name}")

    print("\nModel export complete.")


if __name__ == "__main__":
    main()
