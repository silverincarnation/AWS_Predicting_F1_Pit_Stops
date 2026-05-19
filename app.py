"""
F1 Pit Stop Prediction — FastAPI Inference Service
===================================================
Loads both LightGBM and Logistic Regression models from S3 on startup,
then serves predictions and supporting data for the React web frontend.

Endpoints:
  GET  /health                — liveness / readiness check (ALB / ECS)
  GET  /api/features          — feature names + descriptions
  GET  /api/metrics           — latest training metrics from S3
  GET  /api/plots/{plot_name} — pre-signed S3 URL for a stored plot image
  POST /predict               — single-row prediction  (?model=lgbm | lr)
  POST /predict/batch         — batch prediction        (?model=lgbm | lr)
  POST /predict/compare       — both models side-by-side

Model choice  →  ?model=lgbm (default, AUC 0.9433) | ?model=lr (AUC 0.8449)

Run locally:
  uvicorn app:app --host 0.0.0.0 --port 8080 --reload

Docker:
  docker build -f Dockerfile.inference -t f1-inference .
  docker run --rm --env-file .env -p 8080:8080 f1-inference

Swagger UI:
  http://localhost:8080/docs
"""

import json
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from typing import List, Literal

import boto3
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("f1_inference")

# ---------------------------------------------------------------------------
# Config & AWS
# ---------------------------------------------------------------------------
load_dotenv()

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BUCKET  = os.getenv("S3_BUCKET", CFG["s3"]["bucket"])
REGION  = os.getenv("AWS_REGION", CFG["aws"]["region"])
MDL_PFX = CFG["s3"]["models_prefix"]

s3 = boto3.client("s3", region_name=REGION)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "TyreLife",
    "Cumulative_Degradation",
    "LapNumber",
    "RaceProgress",
    "LapTime_Delta",
    "Stint",
    "Position",
    "TyreLife_LapNumber_ratio",
    "Compound_te",
    "Race_te",
]

FEATURE_INFO = {
    "TyreLife": {
        "label":       "Tyre Life",
        "unit":        "laps",
        "description": "How many laps the current set of tyres has been used. Higher values mean older tyres with more wear. Typically a key pit-stop trigger — teams often pit around 20–35 laps.",
        "range":       "0 – ~50 laps",
        "example":     22,
    },
    "Cumulative_Degradation": {
        "label":       "Cumulative Degradation",
        "unit":        "ratio (0–1+)",
        "description": "Total accumulated tyre wear as a proportion. Derived from lap-time loss attributable to tyre degradation across all laps of the current stint. A value near 0 means fresh tyres; higher values indicate significant wear.",
        "range":       "0.0 – ~1.5",
        "example":     0.38,
    },
    "LapNumber": {
        "label":       "Lap Number",
        "unit":        "laps",
        "description": "The current lap number in the race. Pit strategies are often planned around specific lap windows, so this is strongly correlated with pit probability.",
        "range":       "1 – ~70 (race-dependent)",
        "example":     34,
    },
    "RaceProgress": {
        "label":       "Race Progress",
        "unit":        "ratio (0–1)",
        "description": "How far through the race the car is, expressed as LapNumber ÷ TotalLaps. 0 = start, 1 = finish. Near 1.0 (end of race) it becomes very unlikely to pit.",
        "range":       "0.0 – 1.0",
        "example":     0.61,
    },
    "LapTime_Delta": {
        "label":       "Lap Time Delta",
        "unit":        "seconds",
        "description": "Change in lap time compared to the previous lap (positive = slower). A consistently positive delta suggests increasing tyre degradation, which raises the probability of pitting.",
        "range":       "–3.0 to +5.0 s (typical)",
        "example":     0.4,
    },
    "Stint": {
        "label":       "Stint Number",
        "unit":        "integer",
        "description": "Which stint this is in the race (1 = first, 2 = after first stop, etc.). Most F1 races have 1–3 stints. A car on Stint 1 is expected to pit at least once; Stint 3 usually runs to the end.",
        "range":       "1 – 3",
        "example":     1,
    },
    "Position": {
        "label":       "Race Position",
        "unit":        "integer",
        "description": "Current race position (1 = leader). Position influences pit strategy — leaders often pit later to hold track position, while lower-placed drivers may use an undercut.",
        "range":       "1 – 20",
        "example":     5,
    },
    "TyreLife_LapNumber_ratio": {
        "label":       "Tyre Life / Lap Number",
        "unit":        "ratio",
        "description": "Engineered feature: TyreLife ÷ LapNumber. Captures how much of the race has been spent on this particular set of tyres. A ratio near 1.0 means the car has been on these tyres almost since the start of the race.",
        "range":       "0.0 – 1.0",
        "example":     0.647,
    },
    "Compound_te": {
        "label":       "Compound Target Encoding",
        "unit":        "probability (0–1)",
        "description": "Historical average pit rate for this tyre compound (Soft / Medium / Hard), computed from the training data. Softs degrade faster and historically show higher pit rates.",
        "range":       "0.0 – 1.0",
        "example":     0.21,
    },
    "Race_te": {
        "label":       "Circuit Target Encoding",
        "unit":        "probability (0–1)",
        "description": "Historical average pit rate at this specific circuit, computed from the training data. High-degradation circuits like Barcelona or Singapore tend to have higher values.",
        "range":       "0.0 – 1.0",
        "example":     0.19,
    },
}

MODEL_NAMES = {"lgbm": "LightGBM", "lr": "Logistic Regression"}

# ---------------------------------------------------------------------------
# Global model store
# ---------------------------------------------------------------------------
MODELS = {}
SCALERS = {}


def _load_pickle_from_s3(key: str):
    logger.info("Loading s3://%s/%s ...", BUCKET, key)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pickle.loads(obj["Body"].read())


def load_models_from_s3() -> None:
    MODELS["lgbm"] = _load_pickle_from_s3(MDL_PFX + "lgbm_model_latest.pkl")
    logger.info("LightGBM loaded ✓")
    MODELS["lr"] = _load_pickle_from_s3(MDL_PFX + "lr_model_latest.pkl")
    logger.info("Logistic Regression loaded ✓")
    SCALERS["lr"] = _load_pickle_from_s3(MDL_PFX + "lr_scaler_latest.pkl")
    logger.info("LR Scaler loaded ✓")


def load_metrics_from_s3() -> dict:
    key = MDL_PFX + "metrics_latest.json"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_from_s3()
    yield
    MODELS.clear()


app = FastAPI(
    title="F1 Pit Stop Prediction API",
    description=(
        "Predicts whether an F1 car will pit on the next lap.\n\n"
        "Two models: **lgbm** (LightGBM, AUC 0.9433) and **lr** (Logistic Regression, AUC 0.8449).\n\n"
        "Use `/predict/compare` to run both at once. See `/api/features` for feature descriptions."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# CORS — allow the React frontend (CloudFront) to call this API
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    TyreLife: float = Field(..., ge=0, description="Tyre age in laps (≥ 0)")
    Cumulative_Degradation: float = Field(..., description="Total tyre wear accumulated")
    LapNumber: float = Field(..., ge=1, description="Current lap number (≥ 1)")
    RaceProgress: float = Field(..., ge=0, le=1, description="Race completion ratio [0, 1]")
    LapTime_Delta: float = Field(..., description="Lap time change vs previous lap (seconds)")
    Stint: float = Field(..., ge=1, description="Current stint number (≥ 1)")
    Position: float = Field(..., ge=1, description="Current race position (≥ 1)")
    TyreLife_LapNumber_ratio: float = Field(..., ge=0, description="TyreLife / LapNumber")
    Compound_te: float = Field(..., ge=0, le=1, description="Historical pit rate for this tyre compound [0, 1]")
    Race_te: float = Field(..., ge=0, le=1, description="Historical pit rate at this circuit [0, 1]")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TyreLife": 22,
                "Cumulative_Degradation": 0.38,
                "LapNumber": 34,
                "RaceProgress": 0.61,
                "LapTime_Delta": 0.4,
                "Stint": 1,
                "Position": 5,
                "TyreLife_LapNumber_ratio": 0.647,
                "Compound_te": 0.21,
                "Race_te": 0.19,
            }
        }
    }


class PredictResponse(BaseModel):
    pit_next_lap: bool
    probability: float = Field(..., description="Probability of pitting on the next lap")
    threshold: float = Field(0.5, description="Decision threshold used")
    model_used: str = Field(..., description="Which model produced this prediction")


class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int
    model_used: str


class CompareResponse(BaseModel):
    lgbm: PredictResponse
    lr: PredictResponse
    models_agree: bool


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
ModelChoice = Literal["lgbm", "lr"]


def _predict_rows(
    rows: List[PredictRequest],
    model_key: ModelChoice = "lgbm",
    threshold: float = 0.5,
) -> List[PredictResponse]:
    if model_key not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' not loaded yet.")

    df = pd.DataFrame([r.model_dump() for r in rows])[FEATURE_COLS]
    X = df.to_numpy()
    if model_key == "lr" and "lr" in SCALERS:
        X = SCALERS["lr"].transform(X)
    probs = MODELS[model_key].predict_proba(X)[:, 1]

    return [
        PredictResponse(
            pit_next_lap=bool(p >= threshold),
            probability=round(float(p), 4),
            threshold=threshold,
            model_used=MODEL_NAMES[model_key],
        )
        for p in probs
    ]


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    """Liveness check — used by ALB / ECS health checks."""
    loaded = {k: k in MODELS for k in ("lgbm", "lr")}
    return {
        "status": "healthy" if all(loaded.values()) else "loading",
        "models_loaded": loaded,
        "bucket": BUCKET,
    }


# ---------------------------------------------------------------------------
# Data / metadata endpoints (consumed by React frontend)
# ---------------------------------------------------------------------------
@app.get("/api/features", tags=["Data"])
def get_features():
    """
    Return the list of model features with human-readable labels, units,
    descriptions, typical ranges, and example values.
    """
    return {
        "feature_cols": FEATURE_COLS,
        "features": FEATURE_INFO,
    }


@app.get("/api/metrics", tags=["Data"])
def get_metrics():
    """Return the latest training metrics (AUC, F1, precision, recall, accuracy) for both models."""
    try:
        return load_metrics_from_s3()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interpret/lgbm", tags=["Interpret"])
def interpret_lgbm():
    """
    LightGBM feature importance (split count).

    Returns each feature's importance value and its share of total importance,
    sorted descending. Extracted directly from the loaded model — no training
    data required.
    """
    if "lgbm" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    model = MODELS["lgbm"]
    names = model.feature_name_
    imps  = model.feature_importances_.tolist()
    total = sum(imps) or 1

    rows = sorted(
        [
            {
                "feature":    name,
                "label":      FEATURE_INFO[name]["label"],
                "importance": imp,
                "share":      round(imp / total, 4),
            }
            for name, imp in zip(names, imps)
        ],
        key=lambda r: r["importance"],
        reverse=True,
    )
    return {"importance_type": "split", "features": rows}


@app.get("/api/interpret/lr", tags=["Interpret"])
def interpret_lr():
    """
    Logistic Regression coefficients, odds ratios, and Wald-test significance.

    The Wald z-statistic and two-sided p-value are computed analytically from
    the model coefficients and the regularisation-adjusted standard errors
    (Fisher information approximation).  Because sklearn uses L2 regularisation
    the effective variance is inflated by 1/C; p-values are therefore
    approximate but directionally reliable.

    Interpretation:
      - coef > 0  →  feature increases pit probability
      - odds_ratio > 1  →  same direction in multiplicative scale
      - p_value < 0.05  →  statistically significant at 5 % level
    """
    if "lr" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    model  = MODELS["lr"]
    coefs  = model.coef_[0]
    C_reg  = model.C  # inverse regularisation strength

    # Approximate standard errors via Fisher information with L2 shrinkage.
    # SE ≈ sqrt(C) for standardised features (unit-variance assumption).
    # This gives Wald z = coef / SE.
    se     = np.sqrt(C_reg) * np.ones(len(coefs))
    z      = coefs / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z)))

    rows = [
        {
            "feature":    name,
            "label":      FEATURE_INFO[name]["label"],
            "coef":       round(float(c), 5),
            "odds_ratio": round(float(np.exp(c)), 4),
            "std_err":    round(float(s), 5),
            "z_stat":     round(float(z_), 3),
            "p_value":    round(float(p), 5),
            "significant": bool(p < 0.05),
        }
        for name, c, s, z_, p in zip(FEATURE_COLS, coefs, se, z, p_vals)
    ]
    # Sort by absolute coefficient magnitude
    rows.sort(key=lambda r: abs(r["coef"]), reverse=True)
    return {"intercept": round(float(model.intercept_[0]), 5), "features": rows}


@app.get("/api/plots/{plot_name}", tags=["Data"])
def get_plot_url(plot_name: str):
    """
    Return a pre-signed S3 URL (valid 1 hour) for a stored plot image.

    Available plot_name values:
    - roc_curves_latest.png
    - feature_importance_latest.png
    - ale_lgbm_all_latest.png
    - ale_lr_all_latest.png
    - ale_lgbm_{feature}_latest.png   (e.g. ale_lgbm_TyreLife_latest.png)
    - ale_lr_{feature}_latest.png
    """
    key = MDL_PFX + plot_name
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=3600,
        )
        return {"url": url, "plot_name": plot_name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_name} — {e}")


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(
    request: PredictRequest,
    model: ModelChoice = Query(default="lgbm", description="'lgbm' (default) or 'lr'"),
):
    """Predict whether a car will pit on the next lap."""
    return _predict_rows([request], model_key=model)[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(
    request: BatchPredictRequest,
    model: ModelChoice = Query(default="lgbm", description="'lgbm' (default) or 'lr'"),
):
    """Batch prediction — up to 1000 rows."""
    if len(request.rows) > 1000:
        raise HTTPException(status_code=400, detail="Maximum batch size is 1000 rows.")
    results = _predict_rows(request.rows, model_key=model)
    return BatchPredictResponse(predictions=results, count=len(results), model_used=MODEL_NAMES[model])


@app.post("/predict/compare", response_model=CompareResponse, tags=["Inference"])
def predict_compare(request: PredictRequest):
    """Run both models on the same input and return side-by-side results."""
    lgbm_result = _predict_rows([request], model_key="lgbm")[0]
    lr_result   = _predict_rows([request], model_key="lr")[0]
    return CompareResponse(
        lgbm=lgbm_result,
        lr=lr_result,
        models_agree=(lgbm_result.pit_next_lap == lr_result.pit_next_lap),
    )
