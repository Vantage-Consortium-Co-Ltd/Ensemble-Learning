"""Script to generate valid Jupyter Notebooks for Week 8 Ensemble Learning."""

import json
from pathlib import Path


def create_colab_runner_notebook(output_path: Path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Week 8 - Ensemble Learning: Pure NumPy Bagging and Boosting (SAMME)\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vantage-Consortium-Co-Ltd/Ensemble-Learning/blob/phuriphat/notebooks/week8_colab_runner.ipynb)\n",
                "\n",
                "## 📌 Project Overview\n",
                "This interactive runner demonstrates pure **NumPy & CuPy** implementations of:\n",
                "1. **BaggingClassifier**: Bootstrap Aggregation with parallel process pool execution, voting, and variance reduction.\n",
                "2. **BoostingClassifier**: Multi-class Boosting using the **SAMME** (Stagewise Additive Modeling using a Multi-class Exponential loss function) algorithm and bias reduction.\n",
                "\n",
                "Tested on the **Iris** benchmark dataset (`antoniopelusi/iris`) with an 80/20 stratified split, supporting both local CPU and NVIDIA A100 GPU acceleration on Google Colab."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🛠️ Cell 1: Colab Environment Setup & Hardware Acceleration\n",
                "Detects runtime environment, checks for GPU acceleration (`nvidia-smi`), clones the repository if needed, and installs dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "\n",
                "# Check if running inside Google Colab\n",
                "is_colab = 'google.colab' in sys.modules or os.path.exists('/content')\n",
                "\n",
                "if is_colab:\n",
                "    print(\"🚀 Detected Google Colab environment.\")\n",
                "    print(\"🔍 Checking NVIDIA GPU Status (A100 / T4)...\\n\")\n",
                "    !nvidia-smi\n",
                "    \n",
                "    print(\"\\n📦 Installing dependencies...\")\n",
                "    !pip install --quiet datasets pyyaml scikit-learn pandas matplotlib tabulate joblib\n",
                "    \n",
                "    repo_path = \"/content/Ensemble-Learning\"\n",
                "    if not os.path.exists(repo_path):\n",
                "        print(\"📥 Cloning repository branch 'phuriphat'...\")\n",
                "        !git clone -b phuriphat https://github.com/Vantage-Consortium-Co-Ltd/Ensemble-Learning.git {repo_path}\n",
                "    \n",
                "    if f\"{repo_path}/src\" not in sys.path:\n",
                "        sys.path.insert(0, f\"{repo_path}/src\")\n",
                "    os.chdir(repo_path)\n",
                "else:\n",
                "    print(\"💻 Running in local development environment (CPU Mode).\")\n",
                "    from pathlib import Path\n",
                "    current = Path(os.getcwd()).resolve()\n",
                "    project_root = None\n",
                "    for candidate in [current, current.parent, current.parent.parent]:\n",
                "        if (candidate / \"src\" / \"ensemble_learning\").exists():\n",
                "            project_root = candidate\n",
                "            break\n",
                "    if project_root is None:\n",
                "        project_root = current\n",
                "    os.chdir(str(project_root))\n",
                "    repo_src = str(project_root / \"src\")\n",
                "    if repo_src not in sys.path:\n",
                "        sys.path.insert(0, repo_src)\n",
                "\n",
                "# Verify compute backend\n",
                "try:\n",
                "    from ensemble_learning.utils.device import get_device_info\n",
                "    info = get_device_info()\n",
                "    print(f\"⚡ Array Backend: {info['backend_module']} | Device: {info['device_name']} | CUDA Available: {info['cuda_available']}\")\n",
                "except Exception as e:\n",
                "    import numpy as np\n",
                "    print(f\"⚡ NumPy fallback active ({np.__version__})\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📊 Cell 2: Dataset Ingestion & Stratified 80/20 Splitting\n",
                "Loads the Iris dataset from Hugging Face Hub (`antoniopelusi/iris`) with caching and performs an 80/20 stratified split into Train (120 samples) and Test (30 samples)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "try:\n",
                "    from ensemble_learning.data.make_dataset import load_iris_data, split_dataset\n",
                "    df = load_iris_data(raw_data_dir=\"data/raw\")\n",
                "    X_train, X_test, y_train, y_test = split_dataset(df, train_ratio=0.8, test_ratio=0.2, random_state=42)\n",
                "except Exception as exc:\n",
                "    print(f\"Import notice: {exc}. Using fallback loader.\")\n",
                "    from datasets import load_dataset\n",
                "    try:\n",
                "        ds = load_dataset(\"antoniopelusi/iris\", split=\"train\")\n",
                "        df = pd.DataFrame(ds)\n",
                "    except Exception:\n",
                "        from sklearn.datasets import load_iris\n",
                "        iris_raw = load_iris(as_frame=True)\n",
                "        df = iris_raw.frame\n",
                "        df.columns = [\"sepal-length\", \"sepal-width\", \"petal-length\", \"petal-width\", \"target\"]\n",
                "    feature_cols = [\"sepal-length\", \"sepal-width\", \"petal-length\", \"petal-width\"]\n",
                "    X = df[feature_cols].values\n",
                "    y = df[\"target\"].values\n",
                "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n",
                "\n",
                "print(f\"✅ Iris Dataset Loaded: {len(df)} total rows\")\n",
                "print(f\"🎯 Train Split: {X_train.shape[0]} samples | Test Split: {X_test.shape[0]} samples\")\n",
                "print(f\"📈 Class Distribution in Train: {dict(zip(*np.unique(y_train, return_counts=True)))}\")\n",
                "print(f\"📉 Class Distribution in Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}\")\n",
                "display(df.head())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 Cell 3: Exploratory Data Analysis & Visualization\n",
                "Examine feature distributions across species (`Setosa`, `Versicolor`, `Virginica`) and explore class separability via pairwise distributions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "\n",
                "species_labels = [\"Iris-setosa\", \"Iris-versicolor\", \"Iris-virginica\"]\n",
                "colors = [\"#1f77b4\", \"#2ca02c\", \"#d62728\"]\n",
                "feature_cols = [\"sepal-length\", \"sepal-width\", \"petal-length\", \"petal-width\"]\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
                "for idx, (ax, col) in enumerate(zip(axes.flatten(), feature_cols)):\n",
                "    for class_id, (sp_name, color) in enumerate(zip(species_labels, colors)):\n",
                "        vals = df[df[\"target\"] == class_id][col]\n",
                "        ax.hist(vals, bins=12, alpha=0.6, label=sp_name, color=color, edgecolor=\"black\")\n",
                "    ax.set_title(f\"Distribution: {col}\", fontsize=11, fontweight=\"bold\")\n",
                "    ax.set_xlabel(\"cm\")\n",
                "    ax.set_ylabel(\"Frequency\")\n",
                "    ax.grid(alpha=0.3)\n",
                "    if idx == 0:\n",
                "        ax.legend()\n",
                "\n",
                "plt.suptitle(\"Iris Feature Distributions by Species\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Petal Length vs. Petal Width Scatter\n",
                "plt.figure(figsize=(8, 5))\n",
                "for class_id, (sp_name, color) in enumerate(zip(species_labels, colors)):\n",
                "    subset = df[df[\"target\"] == class_id]\n",
                "    plt.scatter(\n",
                "        subset[\"petal-length\"], subset[\"petal-width\"],\n",
                "        label=sp_name, color=color, s=60, edgecolors=\"black\", alpha=0.85\n",
                "    )\n",
                "plt.title(\"Petal Dimensions: High Class Separability\", fontsize=13, fontweight=\"bold\")\n",
                "plt.xlabel(\"Petal Length (cm)\")\n",
                "plt.ylabel(\"Petal Width (cm)\")\n",
                "plt.grid(alpha=0.3)\n",
                "plt.legend()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🧠 Cell 4: Pure NumPy Bagging and Boosting SAMME Architecture\n",
                "We instantiate and evaluate our pure NumPy ensemble models:\n",
                "- **`BaggingClassifier`**: Generates bootstrap samples $D_1, \\dots, D_M$ with replacement, fits $M$ deep Decision Trees in parallel, and aggregates via majority voting.\n",
                "- **`BoostingClassifier`**: Iteratively trains weak Decision Stumps (depth 1), calculating the multi-class SAMME estimator weight $\\alpha_m = \\log \\frac{1 - \\text{err}_m}{\\text{err}_m} + \\log(K - 1)$ and boosting hard-to-classify samples."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ensemble_learning.models.bagging import BaggingClassifier\n",
                "from ensemble_learning.models.boosting import BoostingClassifier\n",
                "from ensemble_learning.utils.metrics import compute_classification_metrics\n",
                "\n",
                "# 1. Quick test: Bagging with 50 estimators\n",
                "bag_model = BaggingClassifier(n_estimators=50, n_jobs=4, random_state=42)\n",
                "bag_model.fit(X_train, y_train)\n",
                "bag_preds = bag_model.predict(X_test)\n",
                "bag_metrics = compute_classification_metrics(y_test, bag_preds)\n",
                "\n",
                "# 2. Quick test: Boosting (SAMME) with 50 estimators\n",
                "boost_model = BoostingClassifier(n_estimators=50, learning_rate=1.0, random_state=42)\n",
                "boost_model.fit(X_train, y_train)\n",
                "boost_preds = boost_model.predict(X_test)\n",
                "boost_metrics = compute_classification_metrics(y_test, boost_preds)\n",
                "\n",
                "print(\"=== Baseline Ensemble Performance (N=50) ===\")\n",
                "print(f\"🌲 BaggingClassifier  -> Accuracy: {bag_metrics['accuracy']:.4f} | Macro F1: {bag_metrics['f1_macro']:.4f}\")\n",
                "print(f\"🚀 BoostingClassifier -> Accuracy: {boost_metrics['accuracy']:.4f} | Macro F1: {boost_metrics['f1_macro']:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### ⏱️ Cell 5: Benchmark Sweep across $N \\in [10, 50, 100, 500]$\n",
                "We benchmark both ensemble architectures across varying ensemble sizes $N \\in [10, 50, 100, 500]$, measuring:\n",
                "- **Fit Time (s)**: Training wall-clock time\n",
                "- **Inference Latency (s)**: Prediction time on the test set\n",
                "- **Accuracy & Macro F1 Score**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ensemble_learning.models.train import run_benchmark, format_benchmark_table\n",
                "\n",
                "# Execute the benchmark sweep\n",
                "sweep_estimators = [10, 50, 100, 500]\n",
                "print(f\"Running benchmark sweep across N = {sweep_estimators}...\")\n",
                "benchmark_results = run_benchmark(n_estimators_list=sweep_estimators)\n",
                "\n",
                "# Display formatted ASCII table\n",
                "print(\"\\n\" + format_benchmark_table(benchmark_results))\n",
                "\n",
                "# Also convert to pandas DataFrame for interactive viewing\n",
                "benchmark_df = pd.DataFrame(benchmark_results)\n",
                "display(benchmark_df)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📈 Cell 6: Empirical Comparison Charts (Bagging vs. Boosting)\n",
                "Visualizing Accuracy and Training Latency as functions of $N$ estimators:\n",
                "- **Bagging (Variance Reduction)**: Averages independent trees to lower estimation variance without increasing bias. Performance stabilizes quickly as $N$ increases.\n",
                "- **Boosting (Bias Reduction)**: Sequentially focuses on previously misclassified samples. While individual decision stumps have high bias, the linear combination yields low bias and sharp boundaries."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "bag_df = benchmark_df[benchmark_df[\"model\"] == \"Bagging\"].sort_values(\"n_estimators\")\n",
                "boost_df = benchmark_df[benchmark_df[\"model\"] == \"Boosting\"].sort_values(\"n_estimators\")\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Subplot 1: Accuracy vs N\n",
                "ax1.plot(bag_df[\"n_estimators\"], bag_df[\"accuracy\"], marker=\"o\", linewidth=2.5, label=\"Bagging\", color=\"#1f77b4\")\n",
                "ax1.plot(boost_df[\"n_estimators\"], boost_df[\"accuracy\"], marker=\"s\", linewidth=2.5, label=\"Boosting (SAMME)\", color=\"#d62728\")\n",
                "ax1.set_title(\"Test Accuracy vs. Number of Estimators (N)\", fontsize=12, fontweight=\"bold\")\n",
                "ax1.set_xlabel(\"Number of Estimators (N)\")\n",
                "ax1.set_ylabel(\"Test Accuracy\")\n",
                "ax1.set_ylim([0.85, 1.02])\n",
                "ax1.grid(True, alpha=0.3)\n",
                "ax1.legend(loc=\"lower right\")\n",
                "\n",
                "# Subplot 2: Training Latency vs N\n",
                "ax2.plot(bag_df[\"n_estimators\"], bag_df[\"fit_time_sec\"], marker=\"o\", linewidth=2.5, label=\"Bagging (ProcessPool)\", color=\"#1f77b4\")\n",
                "ax2.plot(boost_df[\"n_estimators\"], boost_df[\"fit_time_sec\"], marker=\"s\", linewidth=2.5, label=\"Boosting (Sequential)\", color=\"#d62728\")\n",
                "ax2.set_title(\"Training Latency vs. Number of Estimators (N)\", fontsize=12, fontweight=\"bold\")\n",
                "ax2.set_xlabel(\"Number of Estimators (N)\")\n",
                "ax2.set_ylabel(\"Fit Time (seconds)\")\n",
                "ax2.grid(True, alpha=0.3)\n",
                "ax2.legend(loc=\"upper left\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 💾 Cell 7: Model Artifact Serialization & Export\n",
                "Serializes the best trained models as `.pkl` artifacts and provides download buttons / Google Drive sync options for easy deployment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import joblib\n",
                "from pathlib import Path\n",
                "from ensemble_learning.models.predict import load_and_predict\n",
                "\n",
                "models_dir = Path(\"models\")\n",
                "models_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "bagging_pkl = models_dir / \"bagging_iris.pkl\"\n",
                "boosting_pkl = models_dir / \"boosting_iris.pkl\"\n",
                "\n",
                "# Save models\n",
                "joblib.dump(bag_model, bagging_pkl)\n",
                "joblib.dump(boost_model, boosting_pkl)\n",
                "print(f\"💾 Serialized Bagging model to: {bagging_pkl}\")\n",
                "print(f\"💾 Serialized Boosting model to: {boosting_pkl}\")\n",
                "\n",
                "# Verification with sample inference\n",
                "sample_features = [5.1, 3.5, 1.4, 0.2]  # Expected: 0 (Iris-setosa)\n",
                "pred_bag = load_and_predict(str(bagging_pkl), sample_features)\n",
                "pred_boost = load_and_predict(str(boosting_pkl), sample_features)\n",
                "\n",
                "species_names = [\"Iris-setosa\", \"Iris-versicolor\", \"Iris-virginica\"]\n",
                "print(f\"\\n🔎 Verification Inference on {sample_features}:\")\n",
                "print(f\"   - Bagging Prediction:  Class {pred_bag} ({species_names[pred_bag]})\")\n",
                "print(f\"   - Boosting Prediction: Class {pred_boost} ({species_names[pred_boost]})\")\n",
                "\n",
                "# Download artifacts in Google Colab\n",
                "if is_colab:\n",
                "    try:\n",
                "        from google.colab import files\n",
                "        print(\"\\n⬇️ Triggering browser downloads for serialized models...\")\n",
                "        files.download(str(bagging_pkl))\n",
                "        files.download(str(boosting_pkl))\n",
                "    except Exception as e:\n",
                "        print(f\"Google Colab download notice: {e}\")\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "collapsed_sections": [],
                "name": "week8_colab_runner.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created {output_path}")


def create_eda_and_experiments_notebook(output_path: Path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis & Ensemble Experiments (Iris Dataset)\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vantage-Consortium-Co-Ltd/Ensemble-Learning/blob/phuriphat/notebooks/01_eda_and_experiments.ipynb)\n",
                "\n",
                "## 🎯 Notebook Objectives\n",
                "1. **Exploratory Data Analysis (EDA)** on the Iris dataset: summary statistics, missing values, correlation analysis, and pairwise distributions.\n",
                "2. **Weak Learner Mechanics**: Deep dive into Decision Stump splitting criteria and multi-class SAMME sample weight updates.\n",
                "3. **Decision Boundary Visualizations**: Plotting 2D decision boundary surfaces comparing a Single Decision Tree vs. Bagging vs. Boosting.\n",
                "4. **Variance vs. Bias Trade-off**: Empirical demonstration of how Bagging reduces variance while Boosting reduces bias."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🛠️ Cell 1: Environment Setup & Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Detect environment: Google Colab vs Local Machine (CPU)\n",
                "is_colab = 'google.colab' in sys.modules or os.path.exists('/content')\n",
                "if is_colab:\n",
                "    print(\"🌐 Google Colab environment detected.\")\n",
                "    !pip install --quiet datasets pyyaml scikit-learn pandas matplotlib tabulate joblib\n",
                "    repo_path = \"/content/Ensemble-Learning\"\n",
                "    if not os.path.exists(repo_path):\n",
                "        !git clone -b phuriphat https://github.com/Vantage-Consortium-Co-Ltd/Ensemble-Learning.git {repo_path}\n",
                "    if f\"{repo_path}/src\" not in sys.path:\n",
                "        sys.path.insert(0, f\"{repo_path}/src\")\n",
                "    os.chdir(repo_path)\n",
                "else:\n",
                "    print(\"💻 Local development environment detected (CPU Mode).\")\n",
                "    current = Path(os.getcwd()).resolve()\n",
                "    project_root = None\n",
                "    for candidate in [current, current.parent, current.parent.parent]:\n",
                "        if (candidate / \"src\" / \"ensemble_learning\").exists():\n",
                "            project_root = candidate\n",
                "            break\n",
                "    if project_root is None:\n",
                "        project_root = current\n",
                "    os.chdir(str(project_root))\n",
                "    src_path = str(project_root / \"src\")\n",
                "    if src_path not in sys.path:\n",
                "        sys.path.insert(0, src_path)\n",
                "\n",
                "from ensemble_learning.data.make_dataset import load_iris_data, split_dataset\n",
                "from ensemble_learning.models.bagging import BaggingClassifier\n",
                "from ensemble_learning.models.boosting import BoostingClassifier\n",
                "from ensemble_learning.utils.metrics import compute_classification_metrics\n",
                "from ensemble_learning.utils.device import get_device_info\n",
                "\n",
                "device_info = get_device_info()\n",
                "print(f\"⚡ Array Backend: {device_info['backend_module']} | Device: {device_info['device_name']} (CPU Mode)\")\n",
                "print(f\"🧵 CPU Hardware Cores: {os.cpu_count() or 1}\")\n",
                "print(f\"📁 Working Directory: {os.getcwd()}\")\n",
                "print(\"✅ Environment initialized successfully.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📊 Cell 2: Dataset Ingestion & Statistical Profiling\n",
                "Load dataset from local cache or Hugging Face repository `antoniopelusi/iris`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = load_iris_data(raw_data_dir=\"data/raw\")\n",
                "print(\"Dataset Shape:\", df.shape)\n",
                "print(\"\\nMissing values per column:\")\n",
                "print(df.isnull().sum())\n",
                "\n",
                "print(\"\\nDescriptive Statistics:\")\n",
                "display(df.describe().T)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 Cell 3: Pairwise Feature Relationships & Correlation Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [\"sepal-length\", \"sepal-width\", \"petal-length\", \"petal-width\"]\n",
                "species_labels = [\"Setosa\", \"Versicolor\", \"Virginica\"]\n",
                "colors = [\"#1f77b4\", \"#2ca02c\", \"#d62728\"]\n",
                "\n",
                "# Correlation Matrix\n",
                "corr = df[feature_cols].corr()\n",
                "fig, ax = plt.subplots(figsize=(6, 5))\n",
                "cax = ax.matshow(corr, cmap=\"coolwarm\", vmin=-1, vmax=1)\n",
                "fig.colorbar(cax)\n",
                "ax.set_xticks(range(len(feature_cols)))\n",
                "ax.set_yticks(range(len(feature_cols)))\n",
                "ax.set_xticklabels(feature_cols, rotation=45, ha=\"left\")\n",
                "ax.set_yticklabels(feature_cols)\n",
                "for i in range(len(feature_cols)):\n",
                "    for j in range(len(feature_cols)):\n",
                "        ax.text(j, i, f\"{corr.iloc[i, j]:.2f}\", ha=\"center\", va=\"center\", color=\"black\" if abs(corr.iloc[i, j]) < 0.6 else \"white\")\n",
                "plt.title(\"Feature Correlation Matrix\", pad=20, fontweight=\"bold\")\n",
                "plt.show()\n",
                "\n",
                "# Pairwise Scatter Matrix (4x4)\n",
                "fig, axes = plt.subplots(4, 4, figsize=(14, 14))\n",
                "for i, f1 in enumerate(feature_cols):\n",
                "    for j, f2 in enumerate(feature_cols):\n",
                "        ax = axes[i, j]\n",
                "        if i == j:\n",
                "            for c_id, (sp, col) in enumerate(zip(species_labels, colors)):\n",
                "                vals = df[df[\"target\"] == c_id][f1]\n",
                "                ax.hist(vals, bins=10, alpha=0.5, color=col, edgecolor=\"black\")\n",
                "        else:\n",
                "            for c_id, (sp, col) in enumerate(zip(species_labels, colors)):\n",
                "                sub = df[df[\"target\"] == c_id]\n",
                "                ax.scatter(sub[f2], sub[f1], color=col, alpha=0.7, s=25)\n",
                "        if i == 3:\n",
                "            ax.set_xlabel(f2, fontsize=9)\n",
                "        if j == 0:\n",
                "            ax.set_ylabel(f1, fontsize=9)\n",
                "        ax.grid(alpha=0.2)\n",
                "plt.suptitle(\"Iris 4x4 Pairwise Scatter and Distribution Matrix\", fontsize=14, fontweight=\"bold\", y=1.01)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🌲 Cell 4: Weak Learner Mechanics & Multi-Class SAMME Breakdown\n",
                "In Boosting SAMME (Hastie et al., 2009):\n",
                "1. Each weak learner is a **Decision Stump** (Decision Tree with `max_depth=1`).\n",
                "2. The weighted error is computed as:\n",
                "   $$\\text{err}_m = \\frac{\\sum_{i=1}^N w_i \\cdot \\mathbb{I}(y_i \\neq T_m(x_i))}{\\sum_{i=1}^N w_i}$$\n",
                "3. The multi-class estimator weight $\\alpha_m$ incorporates the number of classes $K$:\n",
                "   $$\\alpha_m = \\log \\frac{1 - \\text{err}_m}{\\text{err}_m} + \\log(K - 1)$$\n",
                "4. Sample weights are updated exponentially:\n",
                "   $$w_i \\leftarrow w_i \\exp\\big(\\alpha_m \\cdot \\mathbb{I}(y_i \\neq T_m(x_i))\\big)$$\n",
                "Let us observe the evolution of $\\alpha_m$ and estimator errors across boosting rounds."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test = split_dataset(df, train_ratio=0.8, test_ratio=0.2, random_state=42)\n",
                "\n",
                "boost = BoostingClassifier(n_estimators=30, learning_rate=1.0, random_state=42)\n",
                "boost.fit(X_train, y_train)\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "ax1.plot(range(1, len(boost.estimator_errors_) + 1), boost.estimator_errors_, marker=\"o\", color=\"#d62728\")\n",
                "ax1.set_title(\"Estimator Weighted Error (err_m)\", fontweight=\"bold\")\n",
                "ax1.set_xlabel(\"Boosting Iteration (m)\")\n",
                "ax1.set_ylabel(\"Weighted Error\")\n",
                "ax1.grid(True, alpha=0.3)\n",
                "\n",
                "ax2.plot(range(1, len(boost.estimator_weights_) + 1), boost.estimator_weights_, marker=\"s\", color=\"#1f77b4\")\n",
                "ax2.set_title(\"Estimator Weight (alpha_m)\", fontweight=\"bold\")\n",
                "ax2.set_xlabel(\"Boosting Iteration (m)\")\n",
                "ax2.set_ylabel(\"Weight alpha_m\")\n",
                "ax2.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🎨 Cell 5: 2D Decision Boundary Comparison (Single Tree vs. Bagging vs. Boosting)\n",
                "We train models on the two most discriminative features: `petal-length` and `petal-width`.\n",
                "Comparing:\n",
                "1. Single Decision Stump (`max_depth=1`)\n",
                "2. Single Decision Tree (`max_depth=3`)\n",
                "3. BaggingClassifier (`n_estimators=50`)\n",
                "4. BoostingClassifier (`n_estimators=50`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.tree import DecisionTreeClassifier\n",
                "\n",
                "# Extract 2 features for 2D visualization\n",
                "X_2d_train = X_train[:, [2, 3]]\n",
                "X_2d_test = X_test[:, [2, 3]]\n",
                "\n",
                "models = {\n",
                "    \"Decision Stump (depth=1)\": DecisionTreeClassifier(max_depth=1, random_state=42),\n",
                "    \"Decision Tree (depth=3)\": DecisionTreeClassifier(max_depth=3, random_state=42),\n",
                "    \"Bagging (50 trees)\": BaggingClassifier(n_estimators=50, n_jobs=-1, random_state=42),\n",
                "    \"Boosting SAMME (50 stumps)\": BoostingClassifier(n_estimators=50, random_state=42),\n",
                "}\n",
                "\n",
                "# Create mesh grid\n",
                "x_min, x_max = X_2d_train[:, 0].min() - 0.5, X_2d_train[:, 0].max() + 0.5\n",
                "y_min, y_max = X_2d_train[:, 1].min() - 0.5, X_2d_train[:, 1].max() + 0.5\n",
                "xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))\n",
                "grid_points = np.c_[xx.ravel(), yy.ravel()]\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 11))\n",
                "cmap_light = [\"#dbe9f6\", \"#d9f0d3\", \"#fcdbd9\"]\n",
                "from matplotlib.colors import ListedColormap\n",
                "cm_bg = ListedColormap(cmap_light)\n",
                "\n",
                "for ax, (name, clf) in zip(axes.flatten(), models.items()):\n",
                "    clf.fit(X_2d_train, y_train)\n",
                "    Z = clf.predict(grid_points).reshape(xx.shape)\n",
                "    test_acc = np.mean(clf.predict(X_2d_test) == y_test)\n",
                "    \n",
                "    ax.contourf(xx, yy, Z, cmap=cm_bg, alpha=0.6)\n",
                "    for c_id, (sp, col) in enumerate(zip(species_labels, colors)):\n",
                "        idx = y_train == c_id\n",
                "        ax.scatter(X_2d_train[idx, 0], X_2d_train[idx, 1], color=col, edgecolors=\"black\", s=40, label=f\"{sp} (Train)\")\n",
                "    ax.set_title(f\"{name}\\nTest Accuracy: {test_acc:.3f}\", fontsize=11, fontweight=\"bold\")\n",
                "    ax.set_xlabel(\"Petal Length (cm)\")\n",
                "    ax.set_ylabel(\"Petal Width (cm)\")\n",
                "    ax.grid(alpha=0.3)\n",
                "    if name == \"Decision Stump (depth=1)\":\n",
                "        ax.legend(loc=\"upper left\", fontsize=8)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📈 Cell 6: Cumulative Estimator Accuracy Convergence\n",
                "We examine how test accuracy evolves as additional estimators are incorporated into the ensemble."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Boosting staged predictions\n",
                "staged_accs = [np.mean(y_stage == y_test) for y_stage in boost.staged_predict(X_test)]\n",
                "\n",
                "plt.figure(figsize=(8, 4.5))\n",
                "plt.plot(range(1, len(staged_accs) + 1), staged_accs, marker=\"o\", color=\"#d62728\", linewidth=2)\n",
                "plt.title(\"Boosting SAMME: Test Accuracy by Iteration Stage\", fontsize=12, fontweight=\"bold\")\n",
                "plt.xlabel(\"Boosting Stage / Estimator Count\")\n",
                "plt.ylabel(\"Test Accuracy\")\n",
                "plt.ylim([0.8, 1.02])\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 💾 Cell 7: Model Export & Summary\n",
                "Serializing final models and recording architectural takeaways:\n",
                "- **Bagging**: Parallelizable, robust to noise and outliers, primarily reduces **variance**.\n",
                "- **Boosting**: Sequential, rapidly drives down training error and **bias**, achieves high margins with minimal base learner depth."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import joblib\n",
                "from pathlib import Path\n",
                "from ensemble_learning.models.predict import load_and_predict\n",
                "\n",
                "out_dir = Path(\"models\")\n",
                "out_dir.mkdir(parents=True, exist_ok=True)\n",
                "model_path = out_dir / \"boosting_eda_final.pkl\"\n",
                "joblib.dump(boost, model_path)\n",
                "print(f\"✅ Serialized trained Boosting model to: {model_path}\")\n",
                "\n",
                "# Verification inference on sample features\n",
                "sample = [5.1, 3.5, 1.4, 0.2]  # Iris-setosa\n",
                "pred = load_and_predict(str(model_path), sample)\n",
                "species_names = [\"Iris-setosa\", \"Iris-versicolor\", \"Iris-virginica\"]\n",
                "print(f\"🔎 Verification inference on {sample}:\")\n",
                "print(f\"   -> Predicted Class {pred} ({species_names[pred]})\")\n",
                "print(\"✅ EDA and experiments workflow complete on local CPU.\")\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created {output_path}")


if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    create_colab_runner_notebook(notebooks_dir / "week8_colab_runner.ipynb")
    create_eda_and_experiments_notebook(notebooks_dir / "01_eda_and_experiments.ipynb")
