"""Metric helpers shared by training, tests, and the written report."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cardiascreen.config import F_BETA


def best_fbeta_threshold(
    y_true,
    proba,
    beta: float = F_BETA,
    low: float = 0.15,
    high: float = 0.85,
    steps: int = 71,
) -> tuple[float, float]:
    """Pick a decision threshold on out-of-fold scores. Never call this on the holdout."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(low, high, steps):
        predicted = (proba >= threshold).astype(int)
        score = float(fbeta_score(y_true, predicted, beta=beta, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def safe_roc_auc(y_true, proba) -> float | None:
    """Skip ROC-AUC when either class has fewer than five rows.

    A slice such as the Switzerland holdout can contain a single negative. The
    AUC on that slice is a coin flip and should not be reported as performance.
    """
    y_true = np.asarray(y_true)
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    if positives < 5 or negatives < 5:
        return None
    return float(roc_auc_score(y_true, proba))


def classification_scores(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    scores = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=F_BETA, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, proba),
        "average_precision": float(average_precision_score(y_true, proba))
        if len(np.unique(y_true)) > 1
        else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positives": int(y_true.sum()),
    }
    return scores


def to_builtin(value):
    """Make a metrics tree safe for json.dump."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
