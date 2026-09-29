"""
QuantumHeal Cloud Inference Engine

Runtime-only implementation.

IMPORTANT:
This file intentionally avoids:
    - PyTorch
    - scikit-learn
    - scikit-image
    - matplotlib

Only NumPy + PennyLane are required.

The model was trained offline and exported into NumPy artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pennylane as qml


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

N_QUBITS = 4
N_LAYERS = 3
IMG_SIZE = 64


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_FILE = ARTIFACT_DIR / "qhs_model.npz"
CNN_FILE = ARTIFACT_DIR / "qhs_cnn.npz"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"


# -------------------------------------------------------------------
# VQC
# -------------------------------------------------------------------

def build_vqc_circuit(
    n_qubits: int = N_QUBITS,
    shots: int | None = None,
):
    """
    Recreates the same circuit topology used during training:

        AngleEmbedding
        BasicEntanglerLayers
        Pauli-Z expectation
    """

    device = qml.device(
        "default.qubit",
        wires=n_qubits,
        shots=shots,
    )

    @qml.qnode(device)
    def circuit(weights, x):
        qml.AngleEmbedding(
            x,
            wires=range(n_qubits),
            rotation="Y",
        )

        qml.BasicEntanglerLayers(
            weights,
            wires=range(n_qubits),
        )

        return qml.expval(
            qml.PauliZ(0)
        )

    return circuit


def vqc_predict(
    weights,
    bias,
    X,
    n_qubits=N_QUBITS,
):
    """
    Exact expectation-value inference.
    """

    circuit = build_vqc_circuit(
        n_qubits=n_qubits,
        shots=None,
    )

    X = np.asarray(X, dtype=float)

    raw = np.array([
        float(
            circuit(weights, x) + bias
        )
        for x in X
    ])

    probs = np.clip(
        (raw + 1.0) / 2.0,
        0.0,
        1.0,
    )

    preds = (
        raw > 0
    ).astype(int)

    return preds, probs


def shot_uncertainty(
    weights,
    bias,
    X,
    n_qubits=N_QUBITS,
    shots=200,
    repeats=10,
):
    """
    Finite-shot uncertainty estimate.
    """

    circuit = build_vqc_circuit(
        n_qubits=n_qubits,
        shots=shots,
    )

    means = []
    stds = []

    for x in np.asarray(X, dtype=float):

        samples = np.array([
            float(
                circuit(weights, x) + bias
            )
            for _ in range(repeats)
        ])

        probs = np.clip(
            (samples + 1.0) / 2.0,
            0.0,
            1.0,
        )

        means.append(
            float(probs.mean())
        )

        stds.append(
            float(probs.std())
        )

    return (
        np.asarray(means),
        np.asarray(stds),
    )


# -------------------------------------------------------------------
# Tabular preprocessing
# -------------------------------------------------------------------

class TabularPreprocessor:

    def __init__(
        self,
        scaler_mean,
        scaler_scale,
        pca_mean,
        pca_components,
        tab_scale,
    ):
        self.scaler_mean = np.asarray(
            scaler_mean,
            dtype=float,
        )

        self.scaler_scale = np.asarray(
            scaler_scale,
            dtype=float,
        )

        self.pca_mean = np.asarray(
            pca_mean,
            dtype=float,
        )

        self.pca_components = np.asarray(
            pca_components,
            dtype=float,
        )

        self.tab_scale = float(
            tab_scale
        )

    def transform(self, X):

        X = np.asarray(
            X,
            dtype=float,
        )

        # StandardScaler.transform
        X_scaled = (
            X - self.scaler_mean
        ) / self.scaler_scale

        # PCA.transform
        X_pca = (
            X_scaled - self.pca_mean
        ) @ self.pca_components.T

        # Same quantum angle scaling used during training
        X_angles = (
            X_pca
            / self.tab_scale
            * np.pi
        )

        return X_angles


# -------------------------------------------------------------------
# NumPy CNN
# -------------------------------------------------------------------

def conv2d_same(
    x,
    weight,
    bias,
):
    """
    NumPy implementation of Conv2d(kernel=3, padding=1, stride=1).

    x:
        (channels, height, width)

    weight:
        (out_channels, in_channels, 3, 3)

    bias:
        (out_channels,)
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    weight = np.asarray(
        weight,
        dtype=np.float32,
    )

    bias = np.asarray(
        bias,
        dtype=np.float32,
    )

    padded = np.pad(
        x,
        (
            (0, 0),
            (1, 1),
            (1, 1),
        ),
        mode="constant",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (3, 3),
        axis=(1, 2),
    )

    # windows:
    # C x H x W x 3 x 3

    out = np.einsum(
        "chwkl,ockl->ohw",
        windows,
        weight,
        optimize=True,
    )

    out += bias[:, None, None]

    return out


def relu(x):
    return np.maximum(
        x,
        0.0,
    )


def max_pool_2x2(x):

    channels, height, width = x.shape

    new_height = height // 2
    new_width = width // 2

    x = x[
        :,
        :new_height * 2,
        :new_width * 2,
    ]

    x = x.reshape(
        channels,
        new_height,
        2,
        new_width,
        2,
    )

    return x.max(
        axis=(2, 4)
    )


def adaptive_avg_pool_4x4(x):

    channels, height, width = x.shape

    block_h = height // 4
    block_w = width // 4

    x = x[
        :,
        :block_h * 4,
        :block_w * 4,
    ]

    x = x.reshape(
        channels,
        4,
        block_h,
        4,
        block_w,
    )

    return x.mean(
        axis=(2, 4)
    )


class NumpyCNN:

    def __init__(
        self,
        arrays,
    ):

        self.conv1_w = arrays["conv1_w"]
        self.conv1_b = arrays["conv1_b"]

        self.conv2_w = arrays["conv2_w"]
        self.conv2_b = arrays["conv2_b"]

        self.conv3_w = arrays["conv3_w"]
        self.conv3_b = arrays["conv3_b"]

        self.embed_w = arrays["embed_w"]
        self.embed_b = arrays["embed_b"]

    def embed(
        self,
        image,
    ):

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        if image.shape != (
            IMG_SIZE,
            IMG_SIZE,
        ):
            raise ValueError(
                f"Image must be "
                f"{IMG_SIZE}x{IMG_SIZE}"
            )

        # ---------------------------------------------------------
        # CNN input
        # ---------------------------------------------------------

        x = image[
            None,
            :,
            :
        ]

        # ---------------------------------------------------------
        # Block 1
        # 64 -> 32
        # ---------------------------------------------------------

        x = conv2d_same(
            x,
            self.conv1_w,
            self.conv1_b,
        )

        x = relu(x)

        x = max_pool_2x2(x)

        # ---------------------------------------------------------
        # Block 2
        # 32 -> 16
        # ---------------------------------------------------------

        x = conv2d_same(
            x,
            self.conv2_w,
            self.conv2_b,
        )

        x = relu(x)

        x = max_pool_2x2(x)

        # ---------------------------------------------------------
        # Block 3
        # 16 -> 16
        # ---------------------------------------------------------

        x = conv2d_same(
            x,
            self.conv3_w,
            self.conv3_b,
        )

        x = relu(x)

        x = adaptive_avg_pool_4x4(x)

        # ---------------------------------------------------------
        # Embedding
        # ---------------------------------------------------------

        flat = x.reshape(-1)

        z = (
            self.embed_w
            @ flat
        ) + self.embed_b

        z = np.tanh(z)

        return z * np.pi


# -------------------------------------------------------------------
# Artifact loader
# -------------------------------------------------------------------

def load_cloud_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}"
        )

    if not CNN_FILE.exists():
        raise FileNotFoundError(
            f"Missing CNN file: {CNN_FILE}"
        )

    model = np.load(
        MODEL_FILE,
        allow_pickle=False,
    )

    cnn_arrays = np.load(
        CNN_FILE,
        allow_pickle=False,
    )

    preprocessor = TabularPreprocessor(
        scaler_mean=model["scaler_mean"],
        scaler_scale=model["scaler_scale"],
        pca_mean=model["pca_mean"],
        pca_components=model["pca_components"],
        tab_scale=model["tab_scale"],
    )

    cnn = NumpyCNN(
        cnn_arrays
    )

    metadata = {}

    if METADATA_FILE.exists():
        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

    return {
        "preprocessor": preprocessor,

        "tab_weights":
            model["tab_weights"],

        "tab_bias":
            float(
                model["tab_bias"]
            ),

        "img_weights":
            model["img_weights"],

        "img_bias":
            float(
                model["img_bias"]
            ),

        "cnn":
            cnn,

        "metadata":
            metadata,
    }

