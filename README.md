# Week 8 - Ensemble Learning: Pure NumPy Bagging and Boosting (SAMME)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vantage-Consortium-Co-Ltd/Ensemble-Learning/blob/phuriphat/notebooks/week8_colab_runner.ipynb)
[![CI Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen.svg)](#testing)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, pure NumPy & CuPy implementation of fundamental **Ensemble Learning** algorithms applied to the multi-class **Iris dataset** (`antoniopelusi/iris` on Hugging Face). Implements **Bagging** with multi-core parallel process execution and **Boosting (SAMME)** multi-class boosting, complete with benchmarking suites, CLI utilities, interactive Jupyter notebooks, and Docker containerization.

---

## 📋 Table of Contents
- [Architecture & Directory Structure](#-architecture--directory-structure)
- [Theoretical Comparison: Bagging vs. Boosting](#-theoretical-comparison-bagging-vs-boosting)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Interactive Notebooks & Google Colab](#-interactive-notebooks--google-colab)
- [Quickstart & Installation](#-quickstart--installation)
- [CLI Usage Guide](#-cli-usage-guide)
- [Docker Containerization](#-docker-containerization)
- [Testing & Quality Assurance](#-testing--quality-assurance)

---

## 🏗️ Architecture & Directory Structure

```text
Week8/
├── configs/
│   └── train_config.yaml             # Hyperparameters, data splits, concurrency & device settings
├── data/
│   ├── raw/iris.csv                  # Cached raw Iris dataset from Hugging Face
│   └── processed/                    # Stratified 80/20 train/test numpy splits
├── models/                           # Serialized joblib model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda_and_experiments.ipynb  # Comprehensive EDA, decision surface meshes & SAMME analysis
│   └── week8_colab_runner.ipynb      # One-click Google Colab runner with GPU acceleration
├── scripts/
│   └── build_notebooks.py            # Reproducible generator for Jupyter Notebook artifacts
├── src/
│   └── ensemble_learning/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── make_dataset.py       # HF dataset streaming, caching & stratified splitting
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py     # Feature matrix extraction & validation
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract BaseClassifier interface
│       │   ├── bagging.py            # Pure NumPy BaggingClassifier with ProcessPoolExecutor
│       │   ├── boosting.py           # Pure NumPy BoostingClassifier (SAMME multi-class)
│       │   ├── train.py              # CLI trainer, matrix benchmarking & serialization
│       │   └── predict.py            # CLI inference runner
│       └── utils/
│           ├── __init__.py
│           ├── config.py             # YAML configuration parser
│           ├── device.py             # NumPy / CuPy (CUDA A100) auto device detection
│           ├── logger.py             # Structured logging utility
│           └── metrics.py            # Accuracy, Macro-F1, Precision, Recall & Timer
├── tests/                            # 55 passing unit & integration tests
│   ├── test_bagging.py
│   ├── test_boosting.py
│   ├── test_config.py
│   ├── test_device.py
│   ├── test_make_dataset.py
│   ├── test_metrics.py
│   ├── test_notebooks.py
│   ├── test_readme_cli.py
│   └── test_train_pipeline.py
├── .dockerignore
├── Dockerfile                        # Multi-stage reproducible container spec (Python 3.12-slim)
├── pyproject.toml                    # PEP 517/518 build metadata & dependencies
└── README.md
```

---

## 🔬 Theoretical Comparison: Bagging vs. Boosting

| Dimension | Bagging (Bootstrap Aggregation) | Boosting (SAMME) |
| :--- | :--- | :--- |
| **Primary Goal** | **Variance Reduction** (Stabilize unstable learners) | **Bias Reduction** (Build strong learners from weak ones) |
| **Base Estimator** | Complex, deep Decision Trees (Low bias, high variance) | Shallow Decision Stumps (`max_depth=1`, High bias, low variance) |
| **Sampling Mechanism** | Independent bootstrap samples drawn with replacement ($N$ samples) | Full dataset reweighted at each stage based on errors |
| **Training Flow** | **Embarrassingly Parallel** (via `ProcessPoolExecutor` / CUDA) | **Sequential** (Round $m$ depends on errors of round $m-1$) |
| **Aggregation Rule** | Unweighted plurality majority voting | Weighted voting $\arg\max_k \sum_{m: T_m(x)=k} \alpha_m$ |
| **Multi-Class Formulation** | Standard majority vote across $K$ classes | SAMME algorithm: $\alpha_m = \log \frac{1 - \text{err}_m}{\text{err}_m} + \log(K - 1)$ |
| **Sensitivity to Noise** | Robust against label noise and outliers | Sensitive to outliers due to exponential sample upweighting |
| **Overfitting Risk** | Low; test error flattens as $N \to \infty$ | Moderate if $N$ is excessively large without regularization |

### Multi-Class SAMME Formulation
For $K$ classes, standard two-class boosting requires base classifiers to achieve $>50\%$ accuracy, which is too stringent for $K \ge 3$. The **SAMME** algorithm generalizes multi-class boosting by modifying the estimator weight update:

$$\alpha_m = \log \left( \frac{1 - \text{err}_m}{\text{err}_m} \right) + \log(K - 1)$$

Sample weights are subsequently updated as:

$$w_i^{(m+1)} = w_i^{(m)} \exp \left( \alpha_m \cdot \mathbb{I}(y_i \neq T_m(x_i)) \right)$$

This ensures that any weak learner performing better than random guessing ($\text{err}_m < 1 - 1/K = 66.7\%$ for Iris where $K=3$) receives a positive weight $\alpha_m > 0$.

---

## 📊 Empirical Benchmark Results

Evaluated on the 80/20 stratified Iris split (120 train samples, 30 test samples):

| Model | Estimators ($N$) | Fit Time (s) | Inference Time (s) | Test Accuracy | Macro F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Boosting (SAMME)** | 10 | **0.0166 s** | 0.0012 s | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| **Boosting (SAMME)** | 50 | 0.0769 s | 0.0085 s | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| **Boosting (SAMME)** | 100 | 0.1859 s | 0.0161 s | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| **Bagging** | 10 | 5.3033 s | 0.0197 s | **0.9667** | **0.9666** | **0.9697** | **0.9667** |
| **Bagging** | 50 | 3.7708 s | 0.0075 s | **0.9667** | **0.9666** | **0.9697** | **0.9667** |
| **Bagging** | 100 | 3.8671 s | 0.0176 s | **0.9667** | **0.9666** | **0.9697** | **0.9667** |

> **Key Takeaway**: Boosting with Decision Stumps trains virtually instantaneously (<0.08 s for $N=50$), attaining 93.3% accuracy. Bagging with parallel Decision Trees achieves 96.67% accuracy by successfully smoothing decision boundaries across overlapping species (Versicolor vs. Virginica).

---

## 🚀 Interactive Notebooks & Google Colab

Run everything with zero local configuration on Google Colab:

| Notebook | Description | Colab Link |
| :--- | :--- | :---: |
| **`week8_colab_runner.ipynb`** | End-to-end interactive runner with GPU auto-detection, benchmark sweeps across $N \in [10, 50, 100, 500]$, accuracy/latency charts, and `.pkl` artifact export. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vantage-Consortium-Co-Ltd/Ensemble-Learning/blob/phuriphat/notebooks/week8_colab_runner.ipynb) |
| **`01_eda_and_experiments.ipynb`** | Deep exploratory data analysis, 4x4 pairwise feature grids, 2D decision boundary mesh visualizations, and SAMME weight convergence curves. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vantage-Consortium-Co-Ltd/Ensemble-Learning/blob/phuriphat/notebooks/01_eda_and_experiments.ipynb) |

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python >= 3.10 (Recommended: Python 3.12)
- Virtual environment (`venv` or `conda`)

### 2. Local Setup
```bash
# Clone the repository
git clone -b phuriphat https://github.com/Vantage-Consortium-Co-Ltd/Ensemble-Learning.git
cd Ensemble-Learning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

For GPU acceleration (NVIDIA CUDA 12.x / Google Colab A100):
```bash
pip install -e ".[gpu]"
```

---

## 🖥️ CLI Usage Guide

The package provides unified command-line interfaces for training, benchmarking, and real-time inference.

### 1. Benchmark Sweep Matrix
Run a comparative hyperparameter sweep over estimator counts ($N \in [10, 50, 100]$):
```bash
python -m ensemble_learning.models.train --sweep
```

Specify custom estimator counts:
```bash
python -m ensemble_learning.models.train --sweep --n-estimators-list 10 50 100 500
```

### 2. Train Single Model Architecture
Train a Bagging model ($N=50$ estimators) and save artifact:
```bash
python -m ensemble_learning.models.train --model bagging --n-estimators 50 --save models/bagging_iris.pkl
```

Train a Boosting (SAMME) model ($N=50$ estimators) and save artifact:
```bash
python -m ensemble_learning.models.train --model boosting --n-estimators 50 --save models/boosting_iris.pkl
```

### 3. Real-Time Inference CLI
Execute prediction using a serialized model artifact by passing 4 numerical measurements (`sepal-length`, `sepal-width`, `petal-length`, `petal-width`):
```bash
python -m ensemble_learning.models.predict --model-path models/bagging_iris.pkl --features 5.1 3.5 1.4 0.2
```

**Output:**
```text
Loading model artifact from models/bagging_iris.pkl...
Running inference for feature vector: [5.1, 3.5, 1.4, 0.2]
Predicted class: 0 (Iris-setosa)
```

---

## 🐳 Docker Containerization

Run the entire pipeline in an isolated, reproducible Docker container.

### Build Docker Image
```bash
docker build -t ensemble-learning:latest .
```

### Run Test Suite in Container
```bash
docker run --rm ensemble-learning:latest pytest tests/ -v
```

### Execute Benchmark Sweep in Container
```bash
docker run --rm ensemble-learning:latest python -m ensemble_learning.models.train --sweep
```

### Mount Host Directory for Model Inference
```bash
# Train and save model into host models directory
docker run --rm -v "${PWD}/models:/app/models" ensemble-learning:latest \
    python -m ensemble_learning.models.train --model boosting --n-estimators 50 --save models/boosting_iris.pkl

# Run inference
docker run --rm -v "${PWD}/models:/app/models" ensemble-learning:latest \
    python -m ensemble_learning.models.predict --model-path models/boosting_iris.pkl --features 6.7 3.0 5.2 2.3
```

---

## 🧪 Testing & Quality Assurance

The codebase contains unit and integration tests covering data loading, feature transformation, Bagging parallelism, Boosting SAMME reweighting, CLI execution, and notebook schema validity.

Run full test suite:
```bash
pytest tests/ -v
```

Run with coverage report:
```bash
pytest --cov=ensemble_learning tests/
```
