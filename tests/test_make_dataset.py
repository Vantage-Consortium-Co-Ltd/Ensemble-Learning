import numpy as np
import pandas as pd
import pytest
from ml_project.data.make_dataset import load_iris_data, split_train_test
from ml_project.features.build_features import extract_features_and_target, standardize_features


def test_load_iris_data_schema(tmp_path):
    df = load_iris_data(raw_data_dir=str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert "target" in df.columns
    # Check 4 feature columns
    feature_cols = [c for c in df.columns if c != "target"]
    assert len(feature_cols) == 4
    # Check local raw cache was saved
    assert (tmp_path / "iris.csv").is_file()


def test_split_train_test_stratified(tmp_path):
    df = load_iris_data(raw_data_dir=str(tmp_path))
    processed_dir = tmp_path / "processed"
    X_train, X_test, y_train, y_test = split_train_test(
        df, train_ratio=0.8, random_state=42, processed_dir=str(processed_dir)
    )
    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert y_train.shape == (120,)
    assert y_test.shape == (30,)
    # Verify stratified balance: 40 train and 10 test per class
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert len(unique_train) == 3
    assert np.all(counts_train == 40)
    assert np.all(counts_test == 10)
    # Verify processed files saved
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "test.csv").is_file()


def test_load_iris_data_offline_fallback(tmp_path, monkeypatch):
    # Simulate HF network failure
    def mock_hf_failure(*args, **kwargs):
        raise ConnectionError("Simulated HF failure")
    monkeypatch.setattr("datasets.load_dataset", mock_hf_failure)
    
    df = load_iris_data(raw_data_dir=str(tmp_path))
    assert len(df) == 150
    assert "target" in df.columns
    assert (tmp_path / "iris.csv").is_file()


def test_extract_features_and_target(tmp_path):
    df = load_iris_data(raw_data_dir=str(tmp_path))
    X, y = extract_features_and_target(df)
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)


def test_standardize_features():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[2.0, 3.0]])
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    assert np.allclose(np.mean(X_train_std, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(X_train_std, axis=0), [1.0, 1.0])
    assert X_test_std.shape == (1, 2)
