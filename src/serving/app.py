"""
FastAPI inference endpoint for battery RUL prediction.

Endpoint: POST /predict
Input:    Last N cycles of a cell (voltage curve stats, capacity, temperature)
Output:   { rul_cycles, lower_bound, upper_bound, confidence_level }

The model loaded here is the best registered model from MLflow Model Registry.
Model version is passed via environment variable — no hardcoded paths.
"""

import os
import logging
from contextlib import asynccontextmanager

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Request / Response schemas ──────────────────────────────────────────────

class CycleFeatures(BaseModel):
    """Feature vector for a single cycle."""
    cycle_index: int
    discharge_capacity: float = Field(..., description="Discharge capacity in Ah")
    soh: float = Field(..., ge=0.0, le=1.0, description="State of Health (0-1)")
    internal_resistance_proxy: float = Field(..., description="Estimated IR in Ohms")
    ica_peak1_height: float
    ica_peak1_position: float
    ica_peak2_height: float
    dva_peak1_height: float
    temperature_mean: float = Field(..., description="Mean temperature during cycle (°C)")


class PredictRequest(BaseModel):
    cell_id: str = Field(..., description="Unique identifier for the cell")
    sequence: list[CycleFeatures] = Field(
        ...,
        min_length=10,
        description="Ordered list of cycle features, most recent last"
    )
    confidence_level: float = Field(default=0.90, ge=0.5, le=0.99)


class PredictResponse(BaseModel):
    cell_id: str
    rul_cycles: float = Field(..., description="Point estimate: cycles remaining to EOL")
    lower_bound: float = Field(..., description="Lower bound of prediction interval")
    upper_bound: float = Field(..., description="Upper bound of prediction interval")
    confidence_level: float
    model_version: str


# ── App & model loading ──────────────────────────────────────────────────────

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_uri = os.environ.get("MODEL_URI", "models:/battery-rul/Production")
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    yield
    model = None

app = FastAPI(
    title="Battery RUL Prediction API",
    description="Predicts remaining useful life of Li-ion battery cells with uncertainty quantification.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # TODO: convert request.sequence -> feature array -> model.predict()
        # TODO: extract uncertainty bounds from model output
        raise NotImplementedError("Prediction logic not yet implemented")
    except NotImplementedError:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for cell {request.cell_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
