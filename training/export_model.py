"""
QuantumHeal — Cloud Model Export

Runs locally during development/training.

Produces framework-independent inference artifacts:

    model_artifacts/qhs_model.npz
    model_artifacts/qhs_cnn.npz
    model_artifacts/metadata.json

The Vercel runtime does NOT require:
    - torch
    - sklearn
    - scikit-image
"""

from pathlib import Path
import sys
import json

import numpy as np
import torch

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

from engine import hybrid_qml_mvp as core

from engine.imaging_arm import (
    load_synthetic_xrays,
    N_QUBITS_IMG,
)

from engine.cnn_embedding import (
    CNNFeaturePipeline,
)


ARTIFACT_DIR = (
    PROJECT_ROOT
    / "model_artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    print("=" * 70)
    print("QuantumHeal — Cloud Model Export")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load dataset
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

    # ---------------------------------------------------------
    # 2. Tabular VQC
    # ---------------------------------------------------------

    print("\n[2/5] Training tabular VQC...")

    X_train, _, scaler, pca = core.preprocess(
        X_train_raw,
        X_train_raw,
    )

    pca_train = pca.transform(
        scaler.transform(
            X_train_raw
        )
    )

    tab_scale = (
        np.max(
            np.abs(pca_train)
        )
        + 1e-9
    )

    tab_arm = core.build_vqc_arm(
        n_qubits=core.N_QUBITS
    )

    tab_weights, tab_bias, _ = (
        tab_arm["train"](
            X_train,
            y_train,
        )
    )

    # ---------------------------------------------------------
    # 3. CNN
    # ---------------------------------------------------------

    print("\n[3/5] Training CNN...")

    images_all = load_synthetic_xrays(
        y,
        seed=core.SEED,
    )

    img_train = (
        images_all[idx_train]
    )

    cnn = CNNFeaturePipeline(
        embed_dim=N_QUBITS_IMG
    )

    cnn.fit_transform(
        img_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 4. Imaging VQC
    # ---------------------------------------------------------

    print("\n[4/5] Training imaging VQC...")

    img_features = cnn.transform(
        img_train
    )

    img_arm = core.build_vqc_arm(
        n_qubits=N_QUBITS_IMG
    )

    img_weights, img_bias, _ = (
        img_arm["train"](
            img_features,
            y_train,
        )
    )

    # ---------------------------------------------------------
    # 5. Export
    # ---------------------------------------------------------

    print("\n[5/5] Exporting lightweight artifacts...")

    # ---------------------------------------------------------
    # Tabular + VQC
    # ---------------------------------------------------------

    np.savez(
        ARTIFACT_DIR / "qhs_model.npz",

        scaler_mean=np.asarray(
            scaler.mean_,
            dtype=np.float64,
        ),

        scaler_scale=np.asarray(
            scaler.scale_,
            dtype=np.float64,
        ),

        pca_mean=np.asarray(
            pca.mean_,
            dtype=np.float64,
        ),

        pca_components=np.asarray(
            pca.components_,
            dtype=np.float64,
        ),

        tab_scale=np.asarray(
            tab_scale,
            dtype=np.float64,
        ),

        tab_weights=np.asarray(
            tab_weights,
            dtype=np.float64,
        ),

        tab_bias=np.asarray(
            tab_bias,
            dtype=np.float64,
        ),

        img_weights=np.asarray(
            img_weights,
            dtype=np.float64,
        ),

        img_bias=np.asarray(
            img_bias,
            dtype=np.float64,
        ),
    )

    # ---------------------------------------------------------
    # CNN — only export what inference needs
    # ---------------------------------------------------------

    state = cnn.model.state_dict()

    np.savez(
        ARTIFACT_DIR / "qhs_cnn.npz",

        conv1_w=
            state[
                "features.0.weight"
            ]
            .detach()
            .cpu()
            .numpy(),

        conv1_b=
            state[
                "features.0.bias"
            ]
            .detach()
            .cpu()
            .numpy(),

        conv2_w=
            state[
                "features.3.weight"
            ]
            .detach()
            .cpu()
            .numpy(),

        conv2_b=
            state[
                "features.3.bias"
            ]
            .detach()
            .cpu()
            .numpy(),

        conv3_w=
            state[
                "features.6.weight"
            ]
            .detach()
            .cpu()
            .numpy(),

        conv3_b=
            state[
                "features.6.bias"
            ]
            .detach()
            .cpu()
            .numpy(),

        embed_w=
            state[
                "embed.weight"
            ]
            .detach()
            .cpu()
            .numpy(),

        embed_b=
            state[
                "embed.bias"
            ]
            .detach()
            .cpu()
            .numpy(),
    )

    metadata = {
        "model_version": "0.4.0-cloud-lightweight",

        "tabular_features":
            int(len(feature_names)),

        "tabular_qubits":
            int(core.N_QUBITS),

        "imaging_qubits":
            int(N_QUBITS_IMG),

        "cnn_embedding_dim":
            int(N_QUBITS_IMG),

        "fusion": "LateFusion",

        "tabular_weight": 0.6,

        "imaging_weight": 0.4,

        "quantum_backend":
            "default.qubit",

        "training_seed":
            int(core.SEED),

        "imaging_dataset":
            "synthetic",
    }

    with open(
        ARTIFACT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nCreated:")

    for file in sorted(
        ARTIFACT_DIR.iterdir()
    ):
        print(
            f"  ✓ {file.name}"
        )

    print(
        "\nCloud export complete."
    )


if __name__ == "__main__":
    main()
