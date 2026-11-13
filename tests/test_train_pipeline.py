"""Integration tests for model training runner, benchmark suite, and prediction CLI."""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from ensemble_learning.models.train import run_benchmark, train_and_save_model
from ensemble_learning.models.predict import load_and_predict


def test_run_benchmark_output_structure():
    """Verify run_benchmark returns required metric keys for all models and configurations."""
    results = run_benchmark(n_estimators_list=[5, 10])
    # Expect 2 models (Bagging, Boosting) * 2 estimator counts = 4 rows
    assert len(results) == 4
    for row in results:
        assert "model" in row
        assert "n_estimators" in row
        assert "fit_time_sec" in row
        assert "inference_time_sec" in row
        assert "accuracy" in row
        assert "f1_macro" in row
        assert "precision_macro" in row
        assert "recall_macro" in row

        assert isinstance(row["fit_time_sec"], float)
        assert row["fit_time_sec"] >= 0.0
        assert isinstance(row["inference_time_sec"], float)
        assert row["inference_time_sec"] >= 0.0
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1_macro"] <= 1.0


def test_train_and_save_model_bagging(tmp_path):
    """Test training and saving a Bagging model, then predicting with saved artifact."""
    model_path = tmp_path / "test_bagging.pkl"
    model = train_and_save_model(
        model_type="bagging",
        n_estimators=10,
        save_path=str(model_path),
    )
    assert model_path.exists()
    assert hasattr(model, "predict")

    # Test predict using saved artifact with list input
    sample = [5.1, 3.5, 1.4, 0.2]  # Iris-setosa
    pred = load_and_predict(str(model_path), sample)
    assert isinstance(pred, int)
    assert pred in [0, 1, 2]


def test_train_and_save_model_boosting(tmp_path):
    """Test training and saving an AdaBoost model, then predicting with saved artifact."""
    model_path = tmp_path / "test_boosting.pkl"
    model = train_and_save_model(
        model_type="boosting",
        n_estimators=10,
        save_path=str(model_path),
    )
    assert model_path.exists()
    assert hasattr(model, "predict")

    # Test predict using saved artifact with numpy array input
    sample_np = np.array([6.7, 3.0, 5.2, 2.3])  # Iris-virginica
    pred = load_and_predict(str(model_path), sample_np)
    assert isinstance(pred, int)
    assert pred in [0, 1, 2]


def test_train_and_save_model_invalid_type():
    """Verify ValueError is raised when an unsupported model_type is provided."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        train_and_save_model(model_type="invalid_model", n_estimators=5)


def test_load_and_predict_input_formats(tmp_path):
    """Verify load_and_predict works with 1D list, 1D array, and 2D array."""
    model_path = tmp_path / "test_model.pkl"
    train_and_save_model(model_type="bagging", n_estimators=5, save_path=str(model_path))

    # 1D list
    pred_list = load_and_predict(str(model_path), [5.0, 3.6, 1.4, 0.2])
    # 1D ndarray
    pred_1d = load_and_predict(str(model_path), np.array([5.0, 3.6, 1.4, 0.2]))
    # 2D ndarray of shape (1, 4)
    pred_2d = load_and_predict(str(model_path), np.array([[5.0, 3.6, 1.4, 0.2]]))

    assert pred_list == pred_1d == pred_2d
    assert isinstance(pred_list, int)


def test_load_and_predict_missing_file():
    """Verify FileNotFoundError is raised if model file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_and_predict("non_existent_model_dir/model.pkl", [5.1, 3.5, 1.4, 0.2])


def test_train_cli_save_and_predict_cli(tmp_path):
    """Test CLI commands for training and prediction using subprocess."""
    model_path = tmp_path / "cli_model.pkl"

    # 1. Run train CLI
    train_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--model",
        "bagging",
        "--n-estimators",
        "5",
        "--save",
        str(model_path),
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
    assert model_path.exists()
    assert "Model saved to" in res_train.stdout or "Saved" in res_train.stdout or "bagging" in res_train.stdout

    # 2. Run predict CLI
    predict_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.predict",
        "--model-path",
        str(model_path),
        "--features",
        "5.1",
        "3.5",
        "1.4",
        "0.2",
    ]
    res_predict = subprocess.run(predict_cmd, capture_output=True, text=True, check=True)
    assert "Predicted class:" in res_predict.stdout or "Prediction:" in res_predict.stdout


def test_train_cli_sweep():
    """Test CLI command for running benchmark sweep."""
    sweep_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--sweep",
        "--n-estimators-list",
        "5",
    ]
    res_sweep = subprocess.run(sweep_cmd, capture_output=True, text=True, check=True)
    assert "Bagging" in res_sweep.stdout
    assert "Boosting" in res_sweep.stdout or "AdaBoost" in res_sweep.stdout
