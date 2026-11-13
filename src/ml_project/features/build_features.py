"""Feature extraction, formatting, and scaling utilities."""

from typing import Optional, Sequence
import numpy as np
import pandas as pd


def extract_features_and_target(
    df: pd.DataFrame,
    target_column: str = "target",
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing feature columns and the target column.
    target_column : str, default="target"
        Name of target column.
    feature_columns : Optional[Sequence[str]], default=None
        List of feature column names. If None, all columns except target_column are used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: np.ndarray of shape (n_samples, n_features), float64
        y: np.ndarray of shape (n_samples,), int64
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[list(feature_columns)].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.int64)
    return X, y


def standardize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Standardize features by zero-centering and scaling to unit variance.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    X_test : Optional[np.ndarray], default=None
        Optional test feature matrix of shape (n_test_samples, n_features) to
        transform using training set statistics.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray]]
        Standardized X_train and optional standardized X_test.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled

    return X_train_scaled, None
