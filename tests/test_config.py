import logging
from pathlib import Path
import pytest

from ml_project.utils.config import load_yaml_config
from ml_project.utils.logger import setup_logger


def test_load_train_config():
    config_path = Path("configs/train_config.yaml")
    cfg = load_yaml_config(str(config_path))
    assert "data" in cfg
    assert cfg["data"]["train_ratio"] == 0.8
    assert cfg["data"]["test_ratio"] == 0.2
    assert "concurrency" in cfg
    assert cfg["concurrency"]["n_jobs"] >= 1


def test_load_model_config():
    config_path = Path("configs/model_config.yaml")
    cfg = load_yaml_config(str(config_path))
    assert "bagging" in cfg
    assert "boosting" in cfg
    assert cfg["bagging"]["max_samples"] == 0.8
    assert cfg["boosting"]["learning_rate"] == 1.0


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("configs/nonexistent_config.yaml")


def test_setup_logger():
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0
