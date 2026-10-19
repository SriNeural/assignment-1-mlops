"""HTTP API for the packaged screening model.

Run with:
    python -m uvicorn cardiascreen.service:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from cardiascreen import __version__
from cardiascreen.config import LOGS_DIR, MODEL_PATH
from cardiascreen.scoring import score_records

REQUESTS = Counter(
    "cardiascreen_requests_total",
    "HTTP requests handled by the screening API.",
    ["endpoint", "status"],
)
PREDICTIONS = Counter(
    "cardiascreen_predictions_total",
    "Screening decisions returned by /predict.",
    ["label", "risk_band"],
)
LATENCY = Histogram(
    "cardiascreen_request_latency_seconds",
    "End-to-end HTTP latency.",
    ["endpoint"],
)
PROBABILITY = Histogram(
    "cardiascreen_predicted_probability",
    "Predicted probability of angiographic heart disease.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

LOGGER = logging.getLogger("cardiascreen.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        return json.dumps(payload)


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    stream = logging.StreamHandler()
    stream.setFormatter(JsonFormatter())
    LOGGER.addHandler(stream)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / "api.jsonl", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        LOGGER.addHandler(file_handler)
    except OSError:
        event = {"event": {"reason": "log directory not writable"}}
        LOGGER.warning("file logging disabled", extra=event)


class PredictRequest(BaseModel):
    """UCI encoding. Chest-pain `cp` is 1-4, not the 0-3 Kaggle recoding."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="UCI chest-pain code 1-4")
    trestbps: float = Field(..., ge=0, le=300, description="Resting BP mm Hg; 0 means unknown")
    chol: float | None = Field(None, ge=0, le=1000, description="Cholesterol mg/dl; 0 is unknown")
    fbs: int | None = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="0 normal, 1 ST-T change, 2 LVH")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-5, le=10, description="Exercise ST depression")
    slope: int = Field(..., ge=1, le=3, description="1 upsloping, 2 flat, 3 downsloping")
    ca: float | None = Field(None, ge=0, le=3, description="Fluoroscopy vessel count, optional")
    thal: float | None = Field(None, description="Thallium code 3, 6, or 7; optional")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: float
    threshold: float
    risk_band: str
    model_name: str
    model_version: str
    request_id: str
    disclaimer: str


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train it with `python -m cardiascreen.train`."
        )
    artifact = joblib.load(path)
    required = {"pipeline", "threshold", "model_name", "model_version"}
    missing = required.difference(artifact)
    if missing:
        raise RuntimeError(f"Artifact at {path} is missing keys: {sorted(missing)}")
    return artifact


def create_app(model_path: str | Path | None = None) -> FastAPI:
    configure_logging()
    path = Path(model_path or os.environ.get("CARDIA_MODEL_PATH", str(MODEL_PATH)))
    artifact = load_artifact(path)
    app = FastAPI(
        title="CardiaScreen",
        version=__version__,
        summary="Binary screening score for angiographic heart disease.",
        description=(
            "The score estimates whether UCI-style clinical fields resemble patients "
            "with angiographic diameter narrowing above 50%. It is a coursework model, "
            "not a diagnosis and not a medical device. Feature codes follow the original "
            "UCI Heart Disease documentation, including chest-pain values 1 through 4."
        ),
    )
    app.state.artifact = artifact
    app.state.model_path = str(path)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            endpoint = request.url.path
            LATENCY.labels(endpoint).observe(elapsed)
            REQUESTS.labels(endpoint, str(status)).inc()

    @app.get("/health")
    def health():
        model = app.state.artifact
        payload = {
            "status": "ok",
            "model_name": model["model_name"],
            "model_version": model["model_version"],
            "threshold": model["threshold"],
        }
        if model.get("data_version"):
            payload["data_version"] = model["data_version"]
        return payload

    @app.get("/metadata")
    def metadata():
        model = app.state.artifact
        return {
            "model_name": model["model_name"],
            "model_version": model["model_version"],
            "trained_at": model.get("trained_at"),
            "threshold": model["threshold"],
            "input_columns": model.get("input_columns"),
            "metrics": model.get("metrics"),
            "disclaimer": model.get("disclaimer"),
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictRequest):
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            decision = score_records(app.state.artifact, [body.model_dump()])[0]
        except Exception as exc:  # surface a client error instead of an empty 500
            LOGGER.exception("score failed", extra={"event": {"request_id": request_id}})
            detail = f"Could not score the request: {exc}"
            raise HTTPException(status_code=400, detail=detail) from exc
        PREDICTIONS.labels(decision["label"], decision["risk_band"]).inc()
        PROBABILITY.observe(decision["probability"])
        payload = {
            **decision,
            "request_id": request_id,
            "disclaimer": app.state.artifact.get("disclaimer", ""),
        }
        LOGGER.info(
            "prediction",
            extra={
                "event": {
                    "request_id": request_id,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "prediction": decision["prediction"],
                    "probability": decision["probability"],
                    "risk_band": decision["risk_band"],
                    "age": body.age,
                    "sex": body.sex,
                    "cp": body.cp,
                }
            },
        )
        return payload

    return app
