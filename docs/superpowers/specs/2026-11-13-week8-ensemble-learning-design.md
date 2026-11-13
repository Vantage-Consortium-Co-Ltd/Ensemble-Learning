# Design Document: Week 8 Ensemble Learning (Bagging vs Boosting from Scratch)

**Date:** 2026-11-13  
**Status:** In Review (Superpowers Brainstorming)  
**Target Module:** Machine Learning (1322308) - Week 8 Assignment  

---

## 1. Intent & The Why

This project fulfills the requirements of the Week 8 Ensemble Learning lab. The objective is to study, implement, and rigorously benchmark two foundational ensemble paradigms:
1. **Bagging (Bootstrap Aggregating):** Variance reduction through parallel training of $N$ diverse models on bootstrap sub-samples with majority voting.
2. **Boosting (Adaptive Boosting / SAMME):** Bias reduction through sequential adaptive learning where misclassified instances receive higher sample weights in subsequent rounds.

To bridge theoretical machine learning concepts with production-grade software engineering, the solution is designed not merely as a single notebook, but as a modular, testable, and configurable Python package (`ml_project/`) compliant with modern ML system standards.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Modular Project Structure:** Deliver a complete `ml_project/` directory structure with configuration management (`configs/`), data caching (`data/`), model serialization (`models/`), core source code (`src/ml_project/`), and automated unit tests (`tests/`).
- **Google Colab A100 GPU Acceleration Support:** Enable high-throughput parallel execution on NVIDIA A100 GPU in Google Colab (Pro/Pro+). Provide GPU tensor acceleration via CuPy (NumPy-compatible GPU array module) and PyTorch CUDA / cuML, offloading compute completely from local machine.
- **CPU vs A100 GPU Benchmark:** Provide comparative benchmarking measuring Execution Latency and Accuracy between Standard CPU execution vs NVIDIA A100 GPU accelerated execution across large ensemble sizes ($N \in \{10, 50, 100, 500\}$).
- **HF Dataset Ingestion:** Automatically load the `antoniopelusi/iris` dataset from Hugging Face Hub (features: `sepal-length`, `sepal-width`, `petal-length`, `petal-width`, and `species` mapped to classes 0, 1, 2), cache raw data locally in `data/raw/`, and perform a stratified 80/20 train/test split.
- **Custom NumPy Bagging Classifier:** Implement bootstrap data sampling and majority voting in pure NumPy. Support parallel execution across worker processes using Python's standard `concurrent.futures.ProcessPoolExecutor`.
- **Custom NumPy Boosting Classifier:** Implement multi-class AdaBoost (SAMME algorithm) in pure NumPy, handling multi-class sample re-weighting, estimator confidence scoring ($\alpha_m$), and weighted voting.
- **Base Classifier:** Use `sklearn.tree.DecisionTreeClassifier` as the base weak/strong estimator.
- **Evaluation & Benchmarking:** Compare Bagging vs. Boosting across key metrics (Accuracy, Precision, Recall, Macro F1) and computational performance (Training time, Inference time) across different ensemble sizes ($N \in \{10, 50, 100\}$).
- **CLI & Artifacts:** Provide CLI entry points (`train.py`, `predict.py`) and save serialized model artifacts (`.pkl`) into `models/`.
- **Automated Verification:** Comprehensive unit tests in `tests/` validating output shapes, voting logic, and sample weight dynamics.

### Non-Goals
- Web UI, REST API, or cloud infrastructure deployment.
- Deep learning ensembles or neural network backbones.
- Writing Decision Tree induction from scratch (the assignment permits library classifiers for the base estimator).

### Anti-Goals
- Direct wrapping or reliance on `sklearn.ensemble.BaggingClassifier` or `sklearn.ensemble.AdaBoostClassifier` (ensemble logic must be written in NumPy from scratch).
- Monolithic spaghetti script placing data loading, algorithms, training, and benchmarking in a single file.

---

## 3. Constraints & Principles

- **Pure NumPy / CuPy Ensemble Core:** Sampling, voting, weight updates, error calculations, and aggregation must use NumPy vectorization, with dynamic CuPy / PyTorch CUDA backend switching when executed on NVIDIA A100 GPU (`device="cuda"`).
- **A100 GPU Optimization & Concurrency:** Concurrency in Bagging must support both CPU multiprocessing (`ProcessPoolExecutor`) and CUDA GPU batch acceleration. When running on A100 GPU (40GB/80GB VRAM, Tensor Cores), array computations and majority voting run directly in GPU memory to maximize throughput.
- **Google Colab Interoperability:** Ensure the codebase, dependencies, and experiments can execute seamlessly on Google Colab with minimal commands (`!pip install datasets pyyaml cupy-cuda12x` / `!pip install -e .` or direct standalone execution in Colab with A100 GPU enabled).
- **Hugging Face First:** Data ingestion pulls `antoniopelusi/iris` from Hugging Face Datasets with fallback to local cache.
- **Multi-class Iris Target:** 3 classes (`Iris-setosa`: 0, `Iris-versicolor`: 1, `Iris-virginica`: 2) and 4 numerical features (`sepal-length`, `sepal-width`, `petal-length`, `petal-width`).

---

## 4. Directory Structure Design (Task 1)

```
ml_project/
├── configs/
│   ├── model_config.yaml         # Hyperparameters (max_depth, learning_rate, criterion)
│   └── train_config.yaml         # Dataset path, split ratio (0.8/0.2), n_estimators, n_jobs
├── data/
│   ├── raw/                      # Hugging Face iris dataset cache (iris.csv)
│   ├── interim/                  # Cleaned/validated intermediate data
│   └── processed/                # Stratified train (80%) and test (20%) datasets
├── models/                       # Serialized trained models (.pkl)
├── notebooks/
│   └── 01_eda_and_experiments.ipynb  # Exploratory Data Analysis & visual plots
├── src/
│   └── ml_project/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── make_dataset.py   # Hugging Face loader, validation, 80/20 train_test_split
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py # Feature scaling / preprocessing if required
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py           # Common estimator interface (fit, predict, score)
│       │   ├── bagging.py        # Custom NumPy Parallel BaggingClassifier
│       │   ├── boosting.py       # Custom NumPy Multi-class AdaBoost (SAMME)
│       │   ├── train.py          # Training pipeline & benchmark execution script
│       │   └── predict.py        # Inference script loading saved .pkl model
│       └── utils/
│           ├── __init__.py
│           ├── metrics.py        # Accuracy, F1, latency timer, confusion matrix
│           └── logger.py         # Standardized console logging
├── tests/
│   ├── __init__.py
│   ├── test_bagging.py           # Unit tests for bootstrap sampling, parallel fit, voting
│   └── test_boosting.py          # Unit tests for SAMME weights, error, multiclass output
├── Dockerfile                    # Containerization definition
├── pyproject.toml                # Project packaging & dependency specs
└── README.md                     # Documentation, usage guide, benchmark findings
```

---

## 5. Algorithmic Specifications (Task 2)

### 5.1 Bagging Classifier (`bagging.py`)
- **Parameters:**
  - `base_estimator`: Class or factory for base classifier (default: `DecisionTreeClassifier(max_depth=None)`).
  - `n_estimators`: Number of models $N$ (e.g. 10, 50, 100).
  - `max_samples`: Float $\le 1.0$ representing subset size (default: 0.8 to satisfy prompt requirement: "จำนวนอาจเท่ากันหรือต่ำกว่า 80% ต้นฉบับ").
  - `bootstrap`: True (sampling with replacement).
  - `n_jobs`: Number of worker processes for `ProcessPoolExecutor`.
  - `random_state`: Integer seed for reproducibility.
- **Fit Procedure:**
  1. For each estimator $k \in \{1, \dots, N\}$, generate bootstrap sample indices:
     $$I_k \sim \text{Uniform}(\{1, \dots, M\}), \quad |I_k| = \lfloor \text{max\_samples} \times M \rfloor$$
  2. Subsample $(X[I_k], y[I_k])$.
  3. Submit training tasks to `ProcessPoolExecutor`.
  4. Collect trained estimator instances.
- **Predict Procedure:**
  1. Collect matrix of individual predictions: $P \in \mathbb{R}^{M_{\text{test}} \times N}$.
  2. Compute majority vote per row using NumPy:
     $$\hat{y}_i = \text{mode}(P_{i, :})$$

### 5.2 Boosting Classifier - Multi-class AdaBoost SAMME (`boosting.py`)
- **Parameters:**
  - `base_estimator`: Base weak learner (default: `DecisionTreeClassifier(max_depth=1)`).
  - `n_estimators`: Number of boosting iterations $T$.
  - `learning_rate`: Shrinkage parameter $\eta \in (0, 1]$ (default: 1.0).
  - `random_state`: Integer seed.
- **Algorithm (SAMME for $K=3$ classes):**
  1. Initialize sample weights uniformly:
     $$w_i^{(1)} = \frac{1}{M}, \quad \forall i \in \{1, \dots, M\}$$
  2. For $m = 1, \dots, T$:
     a. Fit base classifier $h_m(x)$ using weights $w^{(m)}$.
     b. Predict train labels $\hat{y}_i^{(m)} = h_m(x_i)$.
     c. Calculate weighted classification error:
        $$\epsilon_m = \frac{\sum_{i=1}^M w_i^{(m)} \cdot \mathbb{I}(y_i \neq \hat{y}_i^{(m)})}{\sum_{i=1}^M w_i^{(m)}}$$
     d. If $\epsilon_m \ge 1 - \frac{1}{K}$: break early or discard estimator.
     e. If $\epsilon_m = 0$: set high $\alpha_m$ and stop.
     f. Compute estimator weight $\alpha_m$:
        $$\alpha_m = \eta \cdot \left( \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right) + \ln(K - 1) \right)$$
     g. Update sample weights:
        $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(\alpha_m \cdot \mathbb{I}(y_i \neq \hat{y}_i^{(m)})\right)$$
     h. Normalize weights:
        $$w_i^{(m+1)} \leftarrow \frac{w_i^{(m+1)}}{\sum_{j=1}^M w_j^{(m+1)}}$$
  3. **Prediction:**
     For an input $x$, compute class scores across all $K$ classes:
     $$C_k(x) = \sum_{m: h_m(x) = k} \alpha_m, \quad k \in \{0, 1, \dots, K-1\}$$
     $$\hat{y}(x) = \arg\max_{k \in \{0, \dots, K-1\}} C_k(x)$$

---

## 6. Benchmarking & Experiments

The execution runner (`train.py`) will evaluate:
- **Configurations:**
  - Bagging with $N \in [10, 50, 100]$, `max_samples` = 0.8, base `DecisionTreeClassifier(max_depth=None)`.
  - Boosting (AdaBoost SAMME) with $N \in [10, 50, 100]$, `learning_rate` = 1.0, base `DecisionTreeClassifier(max_depth=1)`.
- **Metrics Measured:**
  - Fit Time (ms / seconds)
  - Inference Latency on test set (ms)
  - Test Accuracy (%)
  - Test Macro F1 Score
- **Console Output:** Formatted ASCII/Markdown summary table displayed to stdout.
- **Model Persistence:** Best or final models saved to `models/bagging_iris.pkl` and `models/boosting_iris.pkl`.

---

## 7. What's Left to the Builder

- Exact argument naming in `argparse` for CLI commands.
- Specific random state seed defaults (e.g. `random_state=42`).
- Pytest assertion thresholds and edge-case test values.
