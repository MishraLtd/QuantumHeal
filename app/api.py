"""
QuantumHeal Cloud API

Inference-only FastAPI service.

Heavy training libraries are intentionally NOT imported here.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    Field,
)

from engine.cloud_inference import (
    IMG_SIZE,
    load_cloud_model,
    shot_uncertainty,
    vqc_predict,
)


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)

audit = logging.getLogger(
    "qhs.audit"
)


AUDIT_SALT = os.environ.get(
    "QHS_AUDIT_SALT",
    "change-me-in-production",
)


N_TAB_FEATURES = 30


STATE = {}


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Loading QuantumHeal cloud artifacts..."
    )

    STATE["model"] = load_cloud_model()

    print(
        "QuantumHeal cloud model loaded."
    )

    yield

    STATE.clear()


# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------

app = FastAPI(
    title="QuantumHealth Sentinel API",
    version="0.4.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------

class PredictRequest(BaseModel):

    patient_id: str = Field(
        ...,
        description=(
            "Opaque patient identifier. "
            "Only a salted hash is logged."
        ),
    )

    biomarkers: Optional[List[float]] = Field(
        None,
        description=(
            "30 Wisconsin diagnostic biomarkers."
        ),
    )

    image: Optional[List[List[float]]] = Field(
        None,
        description=(
            f"{IMG_SIZE}x{IMG_SIZE} grayscale "
            "image with values in [0,1]."
        ),
    )


class PredictResponse(BaseModel):

    prediction: str

    fused_probability: float

    modalities_used: List[str]

    contributions: dict

    arms: dict

    low_confidence: bool

    partial_data_warning: Optional[str]

    disclaimer: str

    backend: str

    execution_time_ms: float


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _check_key(
    x_api_key: Optional[str]
):

    required = os.environ.get(
        "QHS_API_KEY"
    )

    if required and (
        x_api_key != required
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def _fuse(
    tabular_prob=None,
    imaging_prob=None,
):
    """
    Same Phase-1 fusion weights:
        tabular = 0.6
        imaging = 0.4
    """

    present = {}

    if tabular_prob is not None:

        present["tabular"] = (
            float(tabular_prob)
        )

    if imaging_prob is not None:

        present["imaging"] = (
            float(imaging_prob)
        )

    if not present:

        raise ValueError(
            "At least one modality is required."
        )

    weights = {
        "tabular": 0.6,
        "imaging": 0.4,
    }

    total_weight = sum(
        weights[name]
        for name in present
    )

    fused = sum(
        weights[name] * value
        for name, value in present.items()
    ) / total_weight

    contributions = {
        name:
            round(
                weights[name]
                / total_weight,
                3,
            )
        for name in present
    }

    prediction = int(
        fused > 0.5
    )

    return (
        fused,
        prediction,
        contributions,
    )


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/api/health")
def health():

    ready = bool(
        STATE.get("model")
    )

    return {
        "status":
            "ok"
            if ready
            else "loading",

        "model_ready":
            ready,
    }


# -------------------------------------------------------------------
# Model information
# -------------------------------------------------------------------

@app.get("/api/model-info")
def model_info():

    metadata = (
        STATE
        .get("model", {})
        .get("metadata", {})
    )

    return {
        "version":
            app.version,

        "architecture":
            "Tabular VQC + CNN Imaging VQC + Late Fusion",

        "quantum_backend":
            metadata.get(
                "quantum_backend",
                "default.qubit",
            ),

        "model_version":
            metadata.get(
                "model_version",
            ),

        "data_provenance": {

            "tabular":
                "Breast Cancer Wisconsin (Diagnostic), public",

            "imaging":
                "Synthetic X-ray-style images",
        },

        "limitations":
            "Research prototype; not validated for clinical use.",
    }


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictResponse,
)
def predict(
    req: PredictRequest,
    x_api_key: Optional[str] = Header(None),
):

    start = time.perf_counter()

    _check_key(x_api_key)

    if (
        req.biomarkers is None
        and req.image is None
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Provide at least one modality: "
                "biomarkers and/or image"
            ),
        )

    model = STATE.get(
        "model"
    )

    if not model:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    used = []

    probs = {}

    arms = {}

    # ---------------------------------------------------------------
    # Tabular
    # ---------------------------------------------------------------

    if req.biomarkers is not None:

        if len(req.biomarkers) != N_TAB_FEATURES:

            raise HTTPException(
                status_code=422,
                detail=(
                    "biomarkers must have "
                    f"{N_TAB_FEATURES} values"
                ),
            )

        x = np.asarray(
            req.biomarkers,
            dtype=float,
        ).reshape(
            1,
            -1,
        )

        processor = model[
            "preprocessor"
        ]

        z = processor.transform(
            x
        )

        _, p = vqc_predict(
            model["tab_weights"],
            model["tab_bias"],
            z,
        )

        _, sd = shot_uncertainty(
            model["tab_weights"],
            model["tab_bias"],
            z,
            n_qubits=4,
            shots=200,
            repeats=10,
        )

        probability = float(
            p[0]
        )

        uncertainty = float(
            sd[0]
        )

        probs["tabular"] = probability

        used.append(
            "tabular"
        )

        arms["tabular"] = {
            "probability":
                probability,

            "uncertainty_std":
                uncertainty,
        }

    # ---------------------------------------------------------------
    # Imaging
    # ---------------------------------------------------------------

    if req.image is not None:

        image = np.asarray(
            req.image,
            dtype=np.float32,
        )

        if image.shape != (
            IMG_SIZE,
            IMG_SIZE,
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    f"image must be "
                    f"{IMG_SIZE}x{IMG_SIZE}"
                ),
            )

        z = model["cnn"].embed(
            image
        )

        z = z.reshape(
            1,
            -1,
        )

        _, p = vqc_predict(
            model["img_weights"],
            model["img_bias"],
            z,
            n_qubits=4,
        )

        _, sd = shot_uncertainty(
            model["img_weights"],
            model["img_bias"],
            z,
            n_qubits=4,
            shots=200,
            repeats=10,
        )

        probability = float(
            p[0]
        )

        uncertainty = float(
            sd[0]
        )

        probs["imaging"] = probability

        used.append(
            "imaging"
        )

        arms["imaging"] = {
            "probability":
                probability,

            "uncertainty_std":
                uncertainty,
        }

    # ---------------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------------

    fused, prediction, contributions = _fuse(
        tabular_prob=
            probs.get("tabular"),

        imaging_prob=
            probs.get("imaging"),
    )

    max_uncertainty = max(
        arm["uncertainty_std"]
        for arm in arms.values()
    )

    low_confidence = bool(
        max_uncertainty > 0.08
        or abs(fused - 0.5) < 0.10
    )

    # ---------------------------------------------------------------
    # Audit
    # ---------------------------------------------------------------

    pid = hashlib.sha256(
        (
            AUDIT_SALT
            + req.patient_id
        ).encode()
    ).hexdigest()[:16]

    audit.info(
        "predict pid=%s modalities=%s pred=%d",
        pid,
        ",".join(used),
        prediction,
    )

    execution_time_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    missing = [
        modality
        for modality in (
            "tabular",
            "imaging",
        )
        if modality not in used
    ]

    warning = None

    if missing:

        warning = (
            "Operating on partial data: "
            "missing "
            + ", ".join(missing)
        )

    return PredictResponse(

        prediction=(
            "disease-positive"
            if prediction == 1
            else "disease-negative"
        ),

        fused_probability=float(
            fused
        ),

        modalities_used=used,

        contributions=contributions,

        arms=arms,

        low_confidence=low_confidence,

        partial_data_warning=warning,

        disclaimer=(
            "Research prototype for "
            "decision support only; "
            "not a diagnosis."
        ),

        backend="default.qubit",

        execution_time_ms=round(
            execution_time_ms,
            2,
        ),
    )
