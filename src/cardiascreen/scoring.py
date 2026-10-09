"""Turn a fitted artifact into a screening decision.

This module is the single place that defines prediction, confidence, and risk band,
so the HTTP service and the CLI cannot drift apart.
"""

from __future__ import annotations

import pandas as pd

from cardiascreen.config import MODEL_INPUT_COLUMNS


def risk_band(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "very_high"


def positive_probability(pipeline, frame: pd.DataFrame) -> list[float]:
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"Model classes do not include the disease label: {classes}")
    index = classes.index(1)
    proba = pipeline.predict_proba(frame)[:, index]
    return [float(value) for value in proba]


def score_records(artifact: dict, records: list[dict]) -> list[dict]:
    frame = pd.DataFrame(records)
    for column in MODEL_INPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[MODEL_INPUT_COLUMNS]
    threshold = float(artifact["threshold"])
    probabilities = positive_probability(artifact["pipeline"], frame)
    decisions = []
    for probability in probabilities:
        prediction = int(probability >= threshold)
        confidence = probability if prediction == 1 else 1.0 - probability
        label = "disease_signal" if prediction == 1 else "no_disease_signal"
        decisions.append(
            {
                "prediction": prediction,
                "label": label,
                "probability": round(probability, 6),
                "confidence": round(float(confidence), 6),
                "threshold": threshold,
                "risk_band": risk_band(probability),
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
            }
        )
    return decisions
