"""Performance metrics and timing utilities for model evaluation."""

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ensemble_learning.utils.device import to_numpy


def compute_classification_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Compute multi-class classification evaluation metrics.

    Calculates accuracy, macro-averaged precision, recall, and F1 score.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Estimated targets returned by a classifier.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "accuracy": float
        - "precision_macro": float
        - "recall_macro": float
        - "f1_macro": float
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision_macro": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0.0)),
        "recall_macro": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0.0)),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0.0)),
    }


class Timer:
    """Context manager for measuring execution time with high precision."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self._end is not None and self._start is not None:
            return self._end - self._start
        if self._start is not None:
            return time.perf_counter() - self._start
        return 0.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000.0

    def __str__(self) -> str:
        if self.elapsed_seconds >= 1.0:
            return f"{self.elapsed_seconds:.4f} s"
        return f"{self.elapsed_ms:.2f} ms"

    def __repr__(self) -> str:
        return f"<Timer elapsed={str(self)}>"
