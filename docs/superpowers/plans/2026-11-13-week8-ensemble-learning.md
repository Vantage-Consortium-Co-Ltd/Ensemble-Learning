# Week 8 Ensemble Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, reproducible ML package implementing Bagging and Multi-class AdaBoost (SAMME) from scratch using pure NumPy with library base estimators on the Hugging Face Iris dataset, with full Google Colab and NVIDIA A100 GPU acceleration support.

**Architecture:** The system is structured as a production-grade modular package (`src/ml_project`) with clean boundaries across data ingestion, custom NumPy/CuPy ensemble models, CLI runners, and benchmark reporting. Bagging achieves variance reduction through bootstrap sub-sampling ($N$ models) and parallel model execution using Python's standard `concurrent.futures.ProcessPoolExecutor` or CUDA acceleration. Boosting achieves bias reduction through multi-class AdaBoost (SAMME) sample re-weighting across sequential rounds. A device abstraction layer automatically detects NVIDIA A100 GPU (CUDA) to leverage CuPy array acceleration for massive speedups on large ensemble sweeps ($N \in \{10, 50, 100, 500\}$), while maintaining seamless pure-NumPy fallback on CPU.

**Tech Stack:** Python 3.12, NumPy 2.x, CuPy / PyTorch (A100 CUDA backend), Scikit-learn (for `DecisionTreeClassifier` base estimator only), Hugging Face `datasets`, Pandas, PyYAML, Joblib, Pytest, Google Colab A100 GPU environment.

**Spec:** `docs/superpowers/specs/2026-11-13-week8-ensemble-learning-design.md`

---

## Global Constraints

- **Pure NumPy / CuPy Ensemble Core:** Bootstrap sampling, majority voting, weighted error calculations ($\epsilon_m$), model weighting ($\alpha_m$), and sample weight distribution updates ($w_i$) must be implemented using vectorized array operations (NumPy on CPU or CuPy on NVIDIA A100 GPU) without invoking `sklearn.ensemble` classes.
- **Base Classifier Library Access:** The base estimator is restricted to `sklearn.tree.DecisionTreeClassifier` (or `cuml.DecisionTreeClassifier` when RAPIDS is installed on A100).
- **A100 GPU & Hardware Concurrency:** The pipeline must support device targeting (`--device auto|cuda|cpu`). When running on an NVIDIA A100 GPU instance in Google Colab, it detects CUDA capability, initializes GPU array context, and performs high-throughput parallel training/voting. On CPU, it gracefully utilizes `ProcessPoolExecutor(max_workers=n_jobs)`.
- **Google Colab Interoperability:** All code must run smoothly both locally and in Google Colab with A100 GPU via an "Open in Colab" notebook (`notebooks/01_eda_and_experiments.ipynb` and `notebooks/week8_colab_runner.ipynb`), with automated GPU driver check (`!nvidia-smi`) and dependency installation.
- **Dataset Source:** Dataset must be loaded from Hugging Face Hub (`antoniopelusi/iris`) and cached locally in `data/raw/iris.csv`. The `species` string labels (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) must be mapped to integer targets `0, 1, 2`.
- **Data Splitting:** Data split must be 80% train and 20% test using stratified sampling to ensure equal class proportions (40 train / 10 test per class).
- **Python Version & Tooling:** Python 3.12+; tests run with `pytest`.

---

## Review Focus

1. **Hugging Face network failure / offline fallback:** When network access is unavailable, `make_dataset.py` must fall back to the cached `data/raw/iris.csv` or `sklearn.datasets.load_iris` without throwing uncaught exceptions. (Covered in Task 2)
2. **AdaBoost division-by-zero or zero-error edge case ($\epsilon_m = 0$):** When a weak learner achieves 100% accuracy on weighted data, $\epsilon_m = 0$ must not cause `ZeroDivisionError` in $\ln((1-\epsilon)/\epsilon)$; the algorithm must cap $\alpha_m$ cleanly and terminate early. (Covered in Task 5)
3. **AdaBoost worse-than-random weak learner ($\epsilon_m \ge 1 - 1/K$):** For $K=3$, if $\epsilon_m \ge 2/3$, the weak learner is worse than random guessing and $\alpha_m \le 0$; the loop must break early or drop the estimator. (Covered in Task 5)
4. **Bagging majority vote tie-breaker:** When $N$ estimators predict a tie between two classes (e.g. 5 votes class 0, 5 votes class 1), prediction must deterministically break the tie without crashing. (Covered in Task 4)
5. **Single-sample 1D vs 2D inference:** Calling `.predict(X)` with a 1D array of shape `(4,)` or single row `(1, 4)` must handle dimension expansion cleanly and return a 1D prediction array. (Covered in Tasks 4 & 5)

---

## File Structure

```
ml_project/
├── configs/
│   ├── model_config.yaml         # Hyperparameter defaults for Bagging & Boosting
│   └── train_config.yaml         # Data paths, train/test split ratio, n_jobs
├── data/
│   ├── raw/                      # Cached raw iris.csv from Hugging Face
│   ├── interim/                  # Intermediate cleaned/validated data
│   └── processed/                # Stratified train (80%) and test (20%) CSVs
├── models/                       # Serialized trained models (.pkl)
├── notebooks/
│   └── 01_eda_and_experiments.ipynb  # Visual EDA, decision boundaries, comparison graphs
├── src/
│   └── ml_project/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── make_dataset.py   # Download HF dataset, local cache, 80/20 stratified split
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py # Feature extraction and array formatting
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py           # BaseClassifier interface with fit, predict, score
│       │   ├── bagging.py        # Custom NumPy Parallel BaggingClassifier
│       │   ├── boosting.py       # Custom NumPy Multi-class AdaBoost (SAMME)
│       │   ├── train.py          # Benchmark training runner across N in [10, 50, 100]
│       │   └── predict.py        # Inference CLI for loading .pkl and evaluating samples
│       └── utils/
│           ├── __init__.py
│           ├── metrics.py        # Accuracy, precision, recall, macro F1, and timer
│           └── logger.py         # Standard formatted logger
├── tests/
│   ├── __init__.py
│   ├── test_config.py            # Tests for configuration loading and validation
│   ├── test_make_dataset.py      # Tests for HF loading, caching, and stratified 80/20 split
│   ├── test_metrics.py           # Tests for classification metrics and timing utilities
│   ├── test_bagging.py           # Tests for bootstrap sampling, parallel fit, and voting
│   ├── test_boosting.py          # Tests for SAMME error, alpha, weight updates, and multiclass predict
│   └── test_train_pipeline.py    # Tests for end-to-end training and serialization
├── pyproject.toml                # Project packaging & dependencies
└── README.md                     # Documentation, CLI commands, benchmark results
```

---

### Task 1: Project Scaffolding and Configuration Management

**Files:**
- Create: `pyproject.toml`
- Create: `configs/model_config.yaml`
- Create: `configs/train_config.yaml`
- Create: `src/ml_project/__init__.py`
- Create: `src/ml_project/utils/__init__.py`
- Create: `src/ml_project/utils/logger.py`
- Create: `src/ml_project/utils/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: None (root configuration layer)
- Produces: `load_yaml_config(path: str) -> dict` in `src/ml_project/utils/config.py`
- Produces: `setup_logger(name: str) -> logging.Logger` in `src/ml_project/utils/logger.py`

- [ ] **Step 1: Write failing test in `tests/test_config.py`**

```python
import pytest
from pathlib import Path
from ml_project.utils.config import load_yaml_config

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_project'`

- [ ] **Step 3: Implement config files, logger, and `load_yaml_config`**

Create `pyproject.toml` defining the project metadata.
Create `configs/model_config.yaml` and `configs/train_config.yaml`.
Implement `load_yaml_config(path: str) -> dict` in `src/ml_project/utils/config.py` reading YAML via PyYAML.
Implement `setup_logger(name: str) -> logging.Logger` in `src/ml_project/utils/logger.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/ src/ml_project/ tests/test_config.py
git commit -m "feat: scaffold ml_project structure and configuration management"
```

---

### Task 2: Hugging Face Dataset Ingestion & Stratified Splitting

**Files:**
- Create: `src/ml_project/data/__init__.py`
- Create: `src/ml_project/data/make_dataset.py`
- Create: `src/ml_project/features/__init__.py`
- Create: `src/ml_project/features/build_features.py`
- Test: `tests/test_make_dataset.py`

**Interfaces:**
- Consumes: `load_yaml_config` from `src/ml_project/utils/config.py`
- Produces: `load_iris_data(raw_data_dir: str = "data/raw") -> pd.DataFrame`
- Produces: `split_train_test(df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing test in `tests/test_make_dataset.py`**

```python
import numpy as np
import pandas as pd
import pytest
from ml_project.data.make_dataset import load_iris_data, split_train_test

def test_load_iris_data_schema(tmp_path):
    df = load_iris_data(raw_data_dir=str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert "target" in df.columns
    # Check 4 feature columns
    feature_cols = [c for c in df.columns if c != "target"]
    assert len(feature_cols) == 4

def test_split_train_test_stratified(tmp_path):
    df = load_iris_data(raw_data_dir=str(tmp_path))
    X_train, X_test, y_train, y_test = split_train_test(df, train_ratio=0.8, random_state=42)
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

def test_load_iris_data_offline_fallback(tmp_path, monkeypatch):
    # Simulate HF network failure
    def mock_hf_failure(*args, **kwargs):
        raise ConnectionError("Simulated HF failure")
    monkeypatch.setattr("datasets.load_dataset", mock_hf_failure)
    
    df = load_iris_data(raw_data_dir=str(tmp_path))
    assert len(df) == 150
    assert "target" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_make_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_project.data.make_dataset'`

- [ ] **Step 3: Implement `load_iris_data` and `split_train_test`**

In `src/ml_project/data/make_dataset.py`:
- Use `datasets.load_dataset("antoniopelusi/iris")` with try/except fallback to `sklearn.datasets.load_iris` when offline.
- Convert features `['sepal-length', 'sepal-width', 'petal-length', 'petal-width']` to float array and map `species` (`Iris-setosa`: 0, `Iris-versicolor`: 1, `Iris-virginica`: 2) to integer `target` column.
- Cache the DataFrame as CSV in `raw_data_dir/iris.csv`.
- Implement `split_train_test` using stratified sampling: for each class $c \in \{0, 1, 2\}$, take $\lfloor 0.8 \times M_c \rfloor$ for train and remainder for test.
- Save processed train and test sets into `data/processed/train.csv` and `data/processed/test.csv`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_make_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml_project/data/ src/ml_project/features/ tests/test_make_dataset.py
git commit -m "feat: implement Hugging Face iris dataset loader with stratified 80/20 split"
```

---

### Task 3: Base Ensemble Interface, Device Backend (CPU/A100 GPU) & Metrics

**Files:**
- Create: `src/ml_project/models/base.py`
- Create: `src/ml_project/utils/device.py`
- Create: `src/ml_project/utils/metrics.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: None
- Produces: `BaseClassifier` (abstract base class with `fit(X, y)`, `predict(X)`, `score(X, y)`) in `src/ml_project/models/base.py`
- Produces: `get_array_module(device: str = "auto") -> tuple[module, str]` in `src/ml_project/utils/device.py`
- Produces: `compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict` in `src/ml_project/utils/metrics.py`
- Produces: `Timer` context manager measuring elapsed milliseconds/seconds

- [ ] **Step 1: Write failing test in `tests/test_metrics.py` and `tests/test_device.py`**

```python
import numpy as np
import pytest
from ml_project.utils.metrics import compute_classification_metrics, Timer
from ml_project.utils.device import get_array_module, to_numpy
import time

def test_device_fallback_cpu():
    xp, dev_name = get_array_module(device="cpu")
    assert xp.__name__ == "numpy"
    assert dev_name == "cpu"
    arr = xp.array([1, 2, 3])
    np_arr = to_numpy(arr)
    assert isinstance(np_arr, np.ndarray)

def test_compute_classification_metrics_multiclass():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert np.isclose(metrics["accuracy"], 5 / 6)

def test_timer_context_manager():
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed_seconds >= 0.04
    assert t.elapsed_ms >= 40.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `BaseClassifier`, `device.py`, `compute_classification_metrics`, and `Timer`**

In `src/ml_project/utils/device.py`:
- Check for CuPy / PyTorch CUDA support. If available and `device in ["auto", "cuda"]`, return `(cupy, "cuda")`, else return `(numpy, "cpu")`.
- Implement `to_numpy(arr)` converting CuPy arrays back to standard NumPy arrays.
In `src/ml_project/models/base.py`:
- Define `BaseClassifier(ABC)` with abstract methods `fit(X, y)`, `predict(X)`, and concrete `score(X, y)`.
In `src/ml_project/utils/metrics.py`:
- Calculate multi-class accuracy, macro precision, macro recall, and macro F1.
- Implement `Timer` context manager using `time.perf_counter()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py tests/test_device.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml_project/models/base.py src/ml_project/utils/device.py src/ml_project/utils/metrics.py tests/
git commit -m "feat: implement BaseClassifier interface, A100/CPU device backend, and evaluation metrics"
```

---

### Task 4: Pure NumPy Bagging Classifier with Concurrency

**Files:**
- Create: `src/ml_project/models/bagging.py`
- Test: `tests/test_bagging.py`

**Interfaces:**
- Consumes: `BaseClassifier` from `src/ml_project/models/base.py`
- Produces: `BaggingClassifier(base_estimator=None, n_estimators=50, max_samples=0.8, bootstrap=True, n_jobs=4, random_state=42)` in `src/ml_project/models/bagging.py`

- [ ] **Step 1: Write failing test in `tests/test_bagging.py`**

```python
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from ml_project.models.bagging import BaggingClassifier

def test_bagging_bootstrap_sampling():
    model = BaggingClassifier(n_estimators=5, max_samples=0.8, random_state=42)
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model.fit(X, y)
    assert len(model.estimators_) == 5
    for est in model.estimators_:
        assert isinstance(est, DecisionTreeClassifier)

def test_bagging_majority_voting_and_tie_breaking():
    model = BaggingClassifier(n_estimators=4, random_state=42)
    # Manually test internal voting logic on a known prediction matrix
    # Predictions of 4 models on 3 test samples:
    preds = np.array([
        [0, 0, 1, 0], # majority 0
        [1, 1, 2, 2], # tie between 1 and 2 -> deterministic tie break
        [2, 2, 2, 2], # unanimous 2
    ])
    voted = model._majority_vote(preds)
    assert voted[0] == 0
    assert voted[1] in [1, 2]
    assert voted[2] == 2

def test_bagging_single_sample_inference():
    model = BaggingClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(50, 4)
    y = np.random.choice([0, 1, 2], size=50)
    model.fit(X, y)
    # 1D single sample of shape (4,)
    pred_1d = model.predict(X[0])
    assert pred_1d.shape == (1,)
    # 2D single sample of shape (1, 4)
    pred_2d = model.predict(X[0:1])
    assert pred_2d.shape == (1,)
    assert pred_1d[0] == pred_2d[0]

def test_bagging_iris_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    clf = BaggingClassifier(n_estimators=30, max_samples=0.8, n_jobs=2, random_state=42)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = np.mean(preds == y_te)
    assert acc >= 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bagging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_project.models.bagging'`

- [ ] **Step 3: Implement `BaggingClassifier` with `ProcessPoolExecutor`**

In `src/ml_project/models/bagging.py`:
- Inherit from `BaseClassifier`.
- Define module-level worker function `_fit_single_estimator(estimator, X, y, sample_indices)` so it can be serialized and executed across processes without pickling errors.
- In `fit(X, y)`:
  - Generate bootstrap indices using `np.random.choice(n_samples, size=int(max_samples * n_samples), replace=bootstrap)`.
  - Dispatch training of clones of `base_estimator` using `concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)`.
  - Store trained estimators in `self.estimators_`.
- In `predict(X)`:
  - Ensure $X$ is 2D via `np.atleast_2d`.
  - Gather predictions $P_{i, j} = \text{estimator}_j.\text{predict}(X_i)$.
  - Implement `_majority_vote` using `np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bagging.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml_project/models/bagging.py tests/test_bagging.py
git commit -m "feat: implement NumPy BaggingClassifier with parallel execution and voting"
```

---

### Task 5: Pure NumPy Multi-class AdaBoost (SAMME) Classifier

**Files:**
- Create: `src/ml_project/models/boosting.py`
- Test: `tests/test_boosting.py`

**Interfaces:**
- Consumes: `BaseClassifier` from `src/ml_project/models/base.py`
- Produces: `AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)` in `src/ml_project/models/boosting.py`

- [ ] **Step 1: Write failing test in `tests/test_boosting.py`**

```python
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from ml_project.models.boosting import AdaBoostClassifier

def test_boosting_initialization_and_weights():
    clf = AdaBoostClassifier(n_estimators=5, random_state=42)
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1, 2], size=30)
    clf.fit(X, y)
    assert len(clf.estimators_) <= 5
    assert len(clf.estimator_weights_) == len(clf.estimators_)
    assert np.all(clf.estimator_weights_ > 0)

def test_boosting_perfect_weak_learner_early_stop():
    # If weak learner is 100% accurate, error is 0. Should handle zero error without dividing by zero.
    class PerfectLearner:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
            
    X = np.random.randn(20, 2)
    y = np.zeros(20, dtype=int)
    clf = AdaBoostClassifier(base_estimator=PerfectLearner(), n_estimators=10)
    clf.fit(X, y)
    # Must terminate early cleanly without ZeroDivisionError
    assert len(clf.estimators_) == 1

def test_boosting_single_sample_inference():
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1, 2], size=30)
    clf.fit(X, y)
    pred_1d = clf.predict(X[0])
    pred_2d = clf.predict(X[0:1])
    assert pred_1d.shape == (1,)
    assert pred_2d.shape == (1,)
    assert pred_1d[0] == pred_2d[0]

def test_boosting_iris_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = np.mean(preds == y_te)
    assert acc >= 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boosting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_project.models.boosting'`

- [ ] **Step 3: Implement `AdaBoostClassifier` using SAMME in NumPy**

In `src/ml_project/models/boosting.py`:
- Inherit from `BaseClassifier`.
- Initialize `self.classes_ = np.unique(y)`, $K = \text{len}(self.classes\_)$.
- Initialize weights $w_i = \frac{1}{M}$.
- Loop $m = 1 \dots T$:
  - Clone base estimator and fit with `sample_weight=w`.
  - Predict $\hat{y}_m$.
  - Calculate incorrect indicator $I_i = (y_i \neq \hat{y}_{m, i})$.
  - Weighted error $\epsilon_m = \frac{\sum w_i I_i}{\sum w_i}$.
  - Handle edge cases: if $\epsilon_m == 0$, assign $\alpha_m = 1.0$ and stop; if $\epsilon_m \ge 1 - 1/K$, break early.
  - Calculate SAMME confidence: $\alpha_m = \eta \cdot \left( \ln\frac{1 - \epsilon_m}{\epsilon_m} + \ln(K - 1) \right)$.
  - Update weights: $w_i \leftarrow w_i \cdot \exp(\alpha_m \cdot I_i)$.
  - Renormalize: $w_i \leftarrow \frac{w_i}{\sum w_j}$.
- In `predict(X)`:
  - Reshape $X$ to 2D.
  - Initialize scores matrix $S \in \mathbb{R}^{M_{\text{test}} \times K}$.
  - For each $(h_m, \alpha_m)$, compute $\hat{y}_m = h_m.\text{predict}(X)$.
  - Accumulate score $S_{i, k} += \alpha_m$ for each sample where $\hat{y}_{m, i} == k$.
  - Return $\arg\max_{k} S_{i, k}$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boosting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml_project/models/boosting.py tests/test_boosting.py
git commit -m "feat: implement NumPy Multi-class AdaBoost (SAMME) algorithm"
```

---

### Task 6: Training Runner, Benchmarking Matrix, and Model Serialization

**Files:**
- Create: `src/ml_project/models/train.py`
- Create: `src/ml_project/models/predict.py`
- Test: `tests/test_train_pipeline.py`

**Interfaces:**
- Consumes: `make_dataset.py`, `bagging.py`, `boosting.py`, `metrics.py`
- Produces: `run_benchmark(n_estimators_list=[10, 50, 100]) -> list[dict]`
- Produces: Saved model files `models/bagging_iris.pkl` and `models/boosting_iris.pkl`
- Produces: CLI interface in `src/ml_project/models/train.py` and `predict.py`

- [ ] **Step 1: Write failing test in `tests/test_train_pipeline.py`**

```python
import os
import joblib
import pytest
from ml_project.models.train import run_benchmark, train_and_save_model
from ml_project.models.predict import load_and_predict

def test_run_benchmark_output_structure():
    results = run_benchmark(n_estimators_list=[5, 10])
    assert len(results) == 4 # 2 for Bagging, 2 for Boosting
    for row in results:
        assert "model" in row
        assert "n_estimators" in row
        assert "fit_time_sec" in row
        assert "inference_time_sec" in row
        assert "accuracy" in row
        assert "f1_macro" in row

def test_train_and_save_model(tmp_path):
    model_path = tmp_path / "test_bagging.pkl"
    train_and_save_model(model_type="bagging", n_estimators=10, save_path=str(model_path))
    assert model_path.exists()
    
    # Test predict using saved artifact
    sample = [5.1, 3.5, 1.4, 0.2] # Iris-setosa
    pred = load_and_predict(str(model_path), sample)
    assert pred in [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `train.py` and `predict.py`**

In `src/ml_project/models/train.py`:
- Implement `run_benchmark(n_estimators_list=[10, 50, 100])` loading data via `load_iris_data` and `split_train_test`.
- For each $N$, fit Bagging and Boosting, measure `Timer` for fit and predict, compute metrics via `compute_classification_metrics`.
- Print ASCII comparison table.
- Implement CLI with `argparse`:
  `python -m ml_project.models.train --sweep`
  `python -m ml_project.models.train --model bagging --n-estimators 50 --save models/bagging_iris.pkl`
In `src/ml_project/models/predict.py`:
- Implement CLI taking sample input features and model path:
  `python -m ml_project.models.predict --model-path models/bagging_iris.pkl --features 5.1 3.5 1.4 0.2`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml_project/models/train.py src/ml_project/models/predict.py tests/test_train_pipeline.py
git commit -m "feat: implement training benchmark suite, serialization, and predict CLI"
```

---

### Task 7: Google Colab Runner, EDA Notebook & Documentation

**Files:**
- Create: `notebooks/01_eda_and_experiments.ipynb`
- Create: `notebooks/week8_colab_runner.ipynb`
- Create: `README.md`
- Create: `Dockerfile`

**Interfaces:**
- Consumes: `ml_project` package modules
- Produces: Google Colab "Open In Colab" interactive execution notebook
- Produces: Visual plots (Feature pairs, Accuracy vs N, Fit Time vs N) in Jupyter Notebook
- Produces: Complete reproducible documentation and container spec

- [ ] **Step 1: Write execution verification test for README commands**

Verify `python -m ml_project.models.train --sweep` runs successfully from the command line and outputs table.

- [ ] **Step 2: Generate Google Colab Runner Notebooks (`01_eda_and_experiments.ipynb` & `week8_colab_runner.ipynb`)**

Create Colab-ready notebook cells that:
1. **Google Colab & NVIDIA A100 GPU Setup:** Auto-detects Colab environment (`/content`), runs `!nvidia-smi` to report A100 GPU status, VRAM, and CUDA compute capability, and installs dependencies (`!pip install datasets scikit-learn pyyaml pandas matplotlib cupy-cuda12x`).
2. **Dataset Exploration:** Load `antoniopelusi/iris` from Hugging Face Hub, display feature statistics and target distribution.
3. **Data Preprocessing & Splitting:** Execute 80/20 stratified split into Train (120 samples) and Test (30 samples).
4. **NumPy & A100 GPU Ensemble Implementations:** Import `BaggingClassifier` (with parallel CPU process pool or A100 CUDA batch acceleration) and `AdaBoostClassifier` (SAMME multi-class with NumPy/CuPy vectorization).
5. **A100 GPU Benchmark Suite Execution:** Execute $N \in [10, 50, 100, 500]$ benchmark sweep on Colab's A100 GPU resources, completely offloading computation from the local machine.
6. **Visualization & Comparison Charts:**
   - Accuracy vs. $N$ Estimators (Bagging vs Boosting)
   - Training Time vs. $N$ Estimators (A100 GPU vs CPU Latency)
7. **Artifact Export:** Save trained `.pkl` models and provide code to download or mount Google Drive.

- [ ] **Step 3: Create `Dockerfile` and `README.md`**

In `Dockerfile`: Python 3.12 base image, copy requirements, install dependencies, entry point.
In `README.md`:
- "Open in Colab" button and quickstart guide for running on Google Colab with 0 local setup.
- Project overview and structure diagram.
- Local installation instructions (`pip install -e .`).
- CLI usage guides for training, sweeping benchmarks, and running inference.
- Summary table of empirical results (Accuracy, F1, and Training Latency for Bagging vs. Boosting).

- [ ] **Step 4: Verify full test suite and CLI execution**

Run: `pytest tests/ -v`
Run: `python -m ml_project.models.train --sweep`
Expected: All tests PASS, benchmark table displays properly.

- [ ] **Step 5: Commit**

```bash
git add notebooks/ README.md Dockerfile
git commit -m "docs: add Google Colab runner notebooks, README documentation, and Dockerfile"
```
