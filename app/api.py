"""
QuantumHealth Sentinel — REST API  (Phase 3: EHR / hospital-system integration)
=================================================================================
Serves the multimodal hybrid quantum-classical model over HTTP so a hospital
information system / EHR can call it.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # interactive docs at http://localhost:8000/docs

Optional env vars:
    QHS_BACKEND   quantum backend (default.qubit | qiskit.aer | ...), see
                  hybrid_qml_mvp.make_device
    QHS_API_KEY   if set, every /predict call must send header  X-API-Key

Endpoints:
    GET  /health       liveness + model status
    GET  /model-info   architecture, backend, data provenance, limitations
    POST /predict      fused prediction from whichever modalities are supplied
                       (biomarkers and/or image) with per-arm shot-based
                       uncertainty and per-modality contribution

Privacy by design (see COMPLIANCE.md): request payloads are NOT persisted.
The audit log stores only a salted hash of patient_id, timestamp, modalities
used and the outcome — never biomarkers or pixels.

IMPORTANT: research prototype. Imaging arm is trained on SYNTHETIC images and
the tabular arm on the public Wisconsin dataset; not a medical device.
"""

import os
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from engine import hybrid_qml_mvp as core
from engine.imaging_arm import load_synthetic_xrays, IMG_SIZE, N_QUBITS_IMG
from engine.cnn_embedding import CNNFeaturePipeline
from engine.fusion import LateFusion
from engine.explainability import build_shot_uncertainty_fn

logging.basicConfig(level=logging.INFO)
audit = logging.getLogger("qhs.audit")
AUDIT_SALT = os.environ.get("QHS_AUDIT_SALT", "change-me-in-production")
N_TAB_FEATURES = 30
STATE = {}


def _train():
    """Train both arms once at startup (~1 min on CPU)."""
    X, y, names = core.load_data()
    idx_tr, _, y_tr, _ = train_test_split(
        np.arange(len(y)), y, test_size=0.25, random_state=core.SEED, stratify=y)
    images = load_synthetic_xrays(y, seed=core.SEED)

    X_tr_p, _, scaler, pca = core.preprocess(X[idx_tr], X[idx_tr][:2])
    tab_scale = np.max(np.abs(pca.transform(scaler.transform(X[idx_tr])))) + 1e-9
    tab_arm = core.build_vqc_arm(core.N_QUBITS)
    tw, tb, _ = tab_arm["train"](X_tr_p, y_tr)

    cnn = CNNFeaturePipeline(embed_dim=N_QUBITS_IMG)
    Z_tr = cnn.fit_transform(images[idx_tr], y_tr)
    img_arm = core.build_vqc_arm(N_QUBITS_IMG)
    iw, ib, _ = img_arm["train"](Z_tr, y_tr)

    STATE.update(
        feature_names=list(names), scaler=scaler, pca=pca, tab_scale=tab_scale,
        tab_arm=tab_arm, tw=tw, tb=tb, cnn=cnn, img_arm=img_arm, iw=iw, ib=ib,
        fusion=LateFusion(),
        tab_unc=build_shot_uncertainty_fn(core.N_QUBITS, shots=200, n_repeats=10),
        img_unc=build_shot_uncertainty_fn(N_QUBITS_IMG, shots=200, n_repeats=10),
        backend=os.environ.get("QHS_BACKEND", "default.qubit"),
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@asynccontextmanager
async def lifespan(app):
    _train()
    yield


app = FastAPI(title="QuantumHealth Sentinel API", version="0.3.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    patient_id: str = Field(..., description="Opaque ID; only a salted hash is logged")
    biomarkers: Optional[List[float]] = Field(
        None, description=f"{N_TAB_FEATURES} Wisconsin-diagnostic features, dataset column order")
    image: Optional[List[List[float]]] = Field(
        None, description=f"{IMG_SIZE}x{IMG_SIZE} grayscale matrix, values in [0,1]")


class ArmResult(BaseModel):
    probability: float
    uncertainty_std: float


class PredictResponse(BaseModel):
    prediction: str
    fused_probability: float
    modalities_used: List[str]
    contributions: dict
    arms: dict
    low_confidence: bool
    partial_data_warning: Optional[str]
    disclaimer: str


def _check_key(x_api_key):
    required = os.environ.get("QHS_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok" if STATE else "loading", "model_ready": bool(STATE)}


@app.get("/model-info")
def model_info():
    return {
        "version": app.version,
        "architecture": "late-fusion of two VQC arms (tabular 4q, imaging 4q)",
        "quantum_backend": STATE.get("backend"),
        "trained_at": STATE.get("trained_at"),
        "data_provenance": {
            "tabular": "Breast Cancer Wisconsin (Diagnostic), public",
            "imaging": "SYNTHETIC X-ray-style images (no real imaging data yet)",
        },
        "limitations": "Research prototype; not validated for clinical use.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    if req.biomarkers is None and req.image is None:
        raise HTTPException(422, "Provide at least one modality: biomarkers and/or image")

    used, probs, arms = [], {}, {}
    if req.biomarkers is not None:
        if len(req.biomarkers) != N_TAB_FEATURES:
            raise HTTPException(422, f"biomarkers must have {N_TAB_FEATURES} values")
        x = np.array(req.biomarkers, dtype=float).reshape(1, -1)
        z = STATE["pca"].transform(STATE["scaler"].transform(x)) / STATE["tab_scale"] * np.pi
        _, p = STATE["tab_arm"]["predict"](STATE["tw"], STATE["tb"], z)
        _, sd = STATE["tab_unc"](STATE["tw"], STATE["tb"], z)
        probs["tabular"] = p; used.append("tabular")
        arms["tabular"] = {"probability": float(p[0]), "uncertainty_std": float(sd[0])}

    if req.image is not None:
        img = np.array(req.image, dtype=np.float32)
        if img.shape != (IMG_SIZE, IMG_SIZE):
            raise HTTPException(422, f"image must be {IMG_SIZE}x{IMG_SIZE}")
        z = STATE["cnn"].transform(img[None])
        _, p = STATE["img_arm"]["predict"](STATE["iw"], STATE["ib"], z)
        _, sd = STATE["img_unc"](STATE["iw"], STATE["ib"], z)
        probs["imaging"] = p; used.append("imaging")
        arms["imaging"] = {"probability": float(p[0]), "uncertainty_std": float(sd[0])}

    fused, pred, contrib = STATE["fusion"].fuse(
        tabular_prob=probs.get("tabular"), imaging_prob=probs.get("imaging"))
    max_sd = max(a["uncertainty_std"] for a in arms.values())
    low_conf = bool(max_sd > 0.08 or abs(fused[0] - 0.5) < 0.1)

    pid = hashlib.sha256((AUDIT_SALT + req.patient_id).encode()).hexdigest()[:16]
    audit.info("predict pid=%s modalities=%s pred=%d", pid, ",".join(used), pred[0])

    return PredictResponse(
        prediction="disease-positive" if pred[0] == 1 else "disease-negative",
        fused_probability=float(fused[0]),
        modalities_used=used,
        contributions=contrib,
        arms=arms,
        low_confidence=low_conf,
        partial_data_warning=("Operating on partial data: missing "
                              + ", ".join(m for m in ("tabular", "imaging") if m not in used))
        if len(used) < 2 else None,
        disclaimer="Research prototype for decision support only; not a diagnosis.",
    )
