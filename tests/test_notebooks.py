"""Validation tests for Jupyter Notebooks."""

import json
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    "notebook_name",
    ["week8_colab_runner.ipynb", "01_eda_and_experiments.ipynb"],
)
def test_notebooks_exist_and_are_valid_json(notebook_name):
    """Verify that required notebooks exist and adhere to nbformat v4."""
    nb_path = Path("notebooks") / notebook_name
    assert nb_path.exists(), f"Notebook {nb_path} does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data.get("nbformat") == 4
    assert len(nb_data["cells"]) >= 7


def test_colab_runner_contains_colab_badge_and_cells():
    """Verify Colab runner contains Open in Colab badge and essential workflow cells."""
    nb_path = Path("notebooks/week8_colab_runner.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    full_text = "".join(
        "".join(cell.get("source", [])) for cell in nb_data["cells"]
    )

    # Check for Open In Colab badge
    assert "https://colab.research.google.com/assets/colab-badge.svg" in full_text
    assert "Vantage-Consortium-Co-Ltd/Ensemble-Learning" in full_text

    # Check for key workflow steps
    assert "nvidia-smi" in full_text
    assert "antoniopelusi/iris" in full_text
    assert "BaggingClassifier" in full_text
    assert "BoostingClassifier" in full_text
    assert "run_benchmark" in full_text
    assert "load_and_predict" in full_text
