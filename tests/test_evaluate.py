import numpy as np

from cardiascreen.evaluate import best_fbeta_threshold, classification_scores, safe_roc_auc


def test_threshold_search_prefers_recall_when_beta_is_high() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.4, 0.45, 0.6, 0.7, 0.9])
    threshold, score = best_fbeta_threshold(y_true, proba, beta=2)
    assert 0.15 <= threshold <= 0.85
    assert score > 0.5


def test_scores_report_the_requested_threshold() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.6, 0.7, 0.8, 0.4, 0.9])
    scores = classification_scores(y_true, proba, threshold=0.5)
    assert scores["support"] == 10
    assert scores["positives"] == 5
    assert scores["threshold"] == 0.5
    assert scores["roc_auc"] > 0.5


def test_roc_auc_is_withheld_when_a_class_is_tiny() -> None:
    y_true = np.array([1, 1, 1, 1, 1, 0])
    proba = np.linspace(0.2, 0.9, 6)
    assert safe_roc_auc(y_true, proba) is None
