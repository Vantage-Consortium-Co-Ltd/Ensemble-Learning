"""Model interfaces, ensemble implementations, training runner, and prediction utilities."""

from typing import Any

from ensemble_learning.models.bagging import BaggingClassifier
from ensemble_learning.models.base import BaseClassifier
from ensemble_learning.models.boosting import AdaBoostClassifier

__all__ = [
    "BaseClassifier",
    "BaggingClassifier",
    "AdaBoostClassifier",
    "run_benchmark",
    "train_and_save_model",
    "load_and_predict",
]


def __getattr__(name: str) -> Any:
    """Lazy export of training and prediction runners to avoid runpy module collision."""
    if name in ("run_benchmark", "train_and_save_model"):
        from ensemble_learning.models.train import run_benchmark, train_and_save_model

        mapping = {
            "run_benchmark": run_benchmark,
            "train_and_save_model": train_and_save_model,
        }
        return mapping[name]
    if name == "load_and_predict":
        from ensemble_learning.models.predict import load_and_predict

        return load_and_predict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
