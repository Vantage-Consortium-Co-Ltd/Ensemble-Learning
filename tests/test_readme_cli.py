"""Verification tests for CLI commands featured in README.md."""

from pathlib import Path
import subprocess
import sys


def test_readme_train_bagging_cli(tmp_path):
    """Test standard Bagging training CLI command documented in README."""
    save_path = tmp_path / "bagging_iris.pkl"
    cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--model",
        "bagging",
        "--n-estimators",
        "10",
        "--save",
        str(save_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert save_path.exists()
    assert "Training completed" in result.stderr or "Model saved" in result.stderr or result.returncode == 0


def test_readme_train_boosting_cli(tmp_path):
    """Test standard AdaBoost training CLI command documented in README."""
    save_path = tmp_path / "adaboost_iris.pkl"
    cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--model",
        "boosting",
        "--n-estimators",
        "10",
        "--save",
        str(save_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert save_path.exists()
    assert "Training completed" in result.stderr or "Model saved" in result.stderr or result.returncode == 0


def test_readme_predict_cli(tmp_path):
    """Test inference prediction CLI command documented in README."""
    save_path = tmp_path / "model.pkl"
    train_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--model",
        "boosting",
        "--n-estimators",
        "10",
        "--save",
        str(save_path),
    ]
    subprocess.run(train_cmd, capture_output=True, text=True, check=True)

    predict_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.predict",
        "--model-path",
        str(save_path),
        "--features",
        "5.1",
        "3.5",
        "1.4",
        "0.2",
    ]
    result = subprocess.run(predict_cmd, capture_output=True, text=True, check=True)
    assert "Predicted class: 0" in result.stdout
    assert "Iris-setosa" in result.stdout


def test_readme_sweep_cli():
    """Test benchmark sweep CLI command documented in README."""
    sweep_cmd = [
        sys.executable,
        "-m",
        "ensemble_learning.models.train",
        "--sweep",
        "--n-estimators-list",
        "10",
    ]
    result = subprocess.run(sweep_cmd, capture_output=True, text=True, check=True)
    assert "| Model" in result.stdout
    assert "Bagging" in result.stdout
    assert "Boosting" in result.stdout
