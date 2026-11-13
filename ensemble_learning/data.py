"""Dataset loading, splitting and scaling (NumPy only)."""
import numpy as np


def load_iris(path):
    """Read the Iris CSV: 4 numeric features + species label per row.

    Tolerates a header row, an Id column and blank lines.
    Returns (X, y, classes) where y holds integer class indices.
    """
    with open(path) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]                          # header row
    data = np.array(rows, dtype=str)
    X = data[:, -5:-1].astype(float)             # 4 features (skips Id if present)
    classes, y = np.unique(data[:, -1], return_inverse=True)
    return X, y, classes


def stratified_split(y, test_frac, rng):
    """Return (train_idx, test_idx) keeping class proportions."""
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.append(idx[: int(round(len(idx) * test_frac))])
    test = np.concatenate(test)
    train = np.setdiff1d(np.arange(len(y)), test)
    return rng.permutation(train), test


def standardize(X_train, X_test):
    """Z-score using training statistics only."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-12
    return (X_train - mu) / sd, (X_test - mu) / sd
