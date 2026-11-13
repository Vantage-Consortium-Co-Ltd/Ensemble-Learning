"""Model inference and prediction CLI for serialized ensemble models."""

import argparse
from pathlib import Path
from typing import List, Sequence, Union

import joblib
import numpy as np

from ensemble_learning.models.base import BaseClassifier
from ensemble_learning.utils.logger import setup_logger

logger = setup_logger("ensemble_learning.models.predict")

CLASS_NAMES = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica",
}


def load_and_predict(
    model_path: Union[str, Path],
    features: Union[Sequence[float], np.ndarray],
) -> int:
    """Load serialized model artifact and predict target class for a single feature vector.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the saved model file (.pkl).
    features : Union[Sequence[float], np.ndarray]
        Input features for inference (e.g. [5.1, 3.5, 1.4, 0.2]).
        Can be a 1D sequence or a 2D array of shape (1, n_features).

    Returns
    -------
    int
        Predicted integer class label (e.g. 0, 1, or 2).

    Raises
    ------
    FileNotFoundError
        If model_path does not exist on disk.
    ValueError
        If features cannot be reshaped into a valid sample vector.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load serialized classifier
    logger.debug("Loading model artifact from %s", path)
    model: BaseClassifier = joblib.load(path)

    # Process input features
    features_arr = np.asarray(features, dtype=np.float64)

    if features_arr.ndim == 1:
        if features_arr.size == 0:
            raise ValueError("Input feature array cannot be empty.")
        features_arr = features_arr.reshape(1, -1)
    elif features_arr.ndim == 2:
        if features_arr.shape[0] != 1:
            raise ValueError(
                f"load_and_predict expects a single sample, but got shape {features_arr.shape}"
            )
    else:
        raise ValueError(
            f"Expected 1D or 2D feature array, got array with {features_arr.ndim} dimensions."
        )

    # Run prediction
    prediction = model.predict(features_arr)
    predicted_class = int(np.asarray(prediction).ravel()[0])

    logger.debug("Predicted class: %d", predicted_class)
    return predicted_class


def main() -> None:
    """CLI entry point for running inference with a serialized model."""
    parser = argparse.ArgumentParser(
        description="Run inference using a serialized Iris ensemble model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to serialized model artifact (.pkl)",
    )
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        required=True,
        help="Feature values in order: sepal-length sepal-width petal-length petal-width",
    )

    args = parser.parse_args()

    predicted_label = load_and_predict(args.model_path, args.features)
    class_name = CLASS_NAMES.get(predicted_label, "Unknown")
    print(f"Predicted class: {predicted_label} ({class_name})")


if __name__ == "__main__":
    main()
