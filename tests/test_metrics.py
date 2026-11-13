import time
import numpy as np
import pytest
from ensemble_learning.utils.metrics import compute_classification_metrics, Timer


def test_compute_classification_metrics_multiclass():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert np.isclose(metrics["accuracy"], 5 / 6)


def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], 1.0)
    assert np.isclose(metrics["precision_macro"], 1.0)
    assert np.isclose(metrics["recall_macro"], 1.0)
    assert np.isclose(metrics["f1_macro"], 1.0)


def test_compute_classification_metrics_handles_lists():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_timer_context_manager():
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed_seconds >= 0.04
    assert t.elapsed_ms >= 40.0
    assert "ms" in str(t) or "s" in str(t)
