"""Model training runner, hyperparameter benchmarking suite, and model serialization."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ensemble_learning.data.make_dataset import load_dataset_from_config
from ensemble_learning.models.bagging import BaggingClassifier
from ensemble_learning.models.base import BaseClassifier
from ensemble_learning.models.boosting import AdaBoostClassifier
from ensemble_learning.utils.config import load_yaml_config
from ensemble_learning.utils.logger import setup_logger
from ensemble_learning.utils.metrics import Timer, compute_classification_metrics

logger = setup_logger("ensemble_learning.models.train")


def format_benchmark_table(results: List[Dict[str, Any]]) -> str:
    """Format benchmark evaluation matrix into an ASCII table string.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of benchmark result dictionaries containing model metrics.

    Returns
    -------
    str
        Rendered ASCII table.
    """
    if not results:
        return "No benchmark results."

    headers = [
        "Model",
        "n_est",
        "Fit Time (s)",
        "Inf Time (s)",
        "Accuracy",
        "F1 (Macro)",
        "Precision",
        "Recall",
    ]

    rows = []
    for r in results:
        rows.append([
            str(r["model"]),
            str(r["n_estimators"]),
            f"{r['fit_time_sec']:.4f}",
            f"{r['inference_time_sec']:.6f}",
            f"{r['accuracy']:.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['precision_macro']:.4f}",
            f"{r['recall_macro']:.4f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def make_divider(char: str = "-", corner: str = "+") -> str:
        parts = [char * (w + 2) for w in col_widths]
        return corner + corner.join(parts) + corner

    def make_row(values: List[str]) -> str:
        formatted_cells = [f" {val.ljust(col_widths[i])} " for i, val in enumerate(values)]
        return "|" + "|".join(formatted_cells) + "|"

    header_line = make_row(headers)
    divider_line = make_divider("-", "+")
    double_divider = make_divider("=", "+")

    output_lines = [
        divider_line,
        header_line,
        double_divider,
    ]
    for row in rows:
        output_lines.append(make_row(row))
        output_lines.append(divider_line)

    return "\n".join(output_lines)


def train_and_save_model(
    model_type: str = "bagging",
    n_estimators: int = 50,
    save_path: str = "models/bagging_iris.pkl",
    config_path: str = "configs/train_config.yaml",
) -> BaseClassifier:
    """Train ensemble classifier on Iris dataset and serialize artifact to disk.

    Parameters
    ----------
    model_type : str, default="bagging"
        Type of model to train. Supported options: "bagging", "boosting", "adaboost".
    n_estimators : int, default=50
        Number of estimators for the ensemble.
    save_path : str, default="models/bagging_iris.pkl"
        File path where trained model will be serialized with joblib.
    config_path : str, default="configs/train_config.yaml"
        Path to training configuration YAML file.

    Returns
    -------
    BaseClassifier
        The fitted classifier instance.

    Raises
    ------
    ValueError
        If model_type is unsupported.
    """
    model_type_norm = model_type.strip().lower()
    if model_type_norm not in ("bagging", "boosting", "adaboost"):
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Expected 'bagging' or 'boosting'/'adaboost'."
        )

    # 1. Load data
    logger.info("Loading training and test data using config: %s", config_path)
    X_train, X_test, y_train, y_test = load_dataset_from_config(config_path)

    # 2. Extract configuration values
    n_jobs = 4
    random_state = 42
    try:
        config = load_yaml_config(config_path)
        n_jobs = config.get("concurrency", {}).get("n_jobs", 4)
        random_state = config.get("data", {}).get("random_state", 42)
    except Exception as exc:
        logger.warning("Could not read configs from %s (%s). Using defaults.", config_path, exc)

    # 3. Instantiate model
    model: BaseClassifier
    if model_type_norm == "bagging":
        model = BaggingClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    # 4. Train model
    logger.info("Training %s classifier with %d estimators...", model_type, n_estimators)
    with Timer() as t_fit:
        model.fit(X_train, y_train)
    logger.info("Training completed in %s", t_fit)

    # 5. Evaluate on test set
    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)
    logger.info(
        "Evaluation on test set - Accuracy: %.4f | F1-Macro: %.4f | Precision-Macro: %.4f | Recall-Macro: %.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["precision_macro"],
        metrics["recall_macro"],
    )

    # 6. Serialize model artifact
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_file)
    logger.info("Model saved successfully to %s", save_file)

    return model


def run_benchmark(
    n_estimators_list: Optional[List[int]] = None,
    config_path: str = "configs/train_config.yaml",
) -> List[Dict[str, Any]]:
    """Run comprehensive benchmarking comparing Bagging and Boosting across estimator counts.

    Measures fitting time, inference time, accuracy, macro-precision, macro-recall, and macro-F1.
    Prints an ASCII summary table.

    Parameters
    ----------
    n_estimators_list : Optional[List[int]], default=None
        List of estimator numbers to benchmark. If None, reads from config or defaults to [10, 50, 100].
    config_path : str, default="configs/train_config.yaml"
        Path to training configuration YAML file.

    Returns
    -------
    List[Dict[str, Any]]
        List of benchmark metric records for each evaluated combination.
    """
    # 1. Load data
    logger.info("Loading Iris dataset for benchmark evaluation...")
    X_train, X_test, y_train, y_test = load_dataset_from_config(config_path)

    # 2. Extract configuration
    n_jobs = 4
    random_state = 42
    default_sweep = [10, 50, 100]
    try:
        config = load_yaml_config(config_path)
        n_jobs = config.get("concurrency", {}).get("n_jobs", 4)
        random_state = config.get("data", {}).get("random_state", 42)
        default_sweep = config.get("models", {}).get("sweep_n_estimators", default_sweep)
    except Exception as exc:
        logger.warning("Could not read configs from %s (%s). Using defaults.", config_path, exc)

    if n_estimators_list is None:
        n_estimators_list = default_sweep

    results: List[Dict[str, Any]] = []

    logger.info("Starting benchmark sweep over n_estimators=%s", n_estimators_list)

    for n_est in n_estimators_list:
        models_to_test = [
            ("Bagging", BaggingClassifier(n_estimators=n_est, n_jobs=n_jobs, random_state=random_state)),
            ("Boosting", AdaBoostClassifier(n_estimators=n_est, random_state=random_state)),
        ]

        for model_name, model in models_to_test:
            # Measure fit time
            with Timer() as t_fit:
                model.fit(X_train, y_train)
            fit_time = t_fit.elapsed_seconds

            # Measure inference time
            with Timer() as t_pred:
                y_pred = model.predict(X_test)
            inf_time = t_pred.elapsed_seconds

            # Evaluate metrics
            metrics = compute_classification_metrics(y_test, y_pred)

            record = {
                "model": model_name,
                "n_estimators": n_est,
                "fit_time_sec": fit_time,
                "inference_time_sec": inf_time,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            }
            results.append(record)

    # Format and display ASCII summary table
    table_str = format_benchmark_table(results)
    print("\n" + table_str + "\n")

    return results


def main() -> None:
    """Command-line interface entry point for training and benchmarking."""
    parser = argparse.ArgumentParser(
        description="Ensemble Learning Training Runner and Benchmarking Matrix",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep benchmarking Bagging vs Boosting",
    )
    parser.add_argument(
        "--n-estimators-list",
        type=int,
        nargs="+",
        default=None,
        help="List of estimator counts for the sweep (e.g., --n-estimators-list 10 50 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bagging",
        choices=["bagging", "boosting", "adaboost"],
        help="Model architecture to train ('bagging' or 'boosting')",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Number of base estimators for single model training (default: 50)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output path for the serialized model artifact (.pkl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML training configuration file",
    )

    args = parser.parse_args()

    if args.sweep:
        print(f"Running benchmark sweep (config: {args.config})...")
        run_benchmark(
            n_estimators_list=args.n_estimators_list,
            config_path=args.config,
        )
    else:
        save_path = args.save if args.save is not None else f"models/{args.model}_iris.pkl"
        print(f"Training {args.model} with n_estimators={args.n_estimators}...")
        train_and_save_model(
            model_type=args.model,
            n_estimators=args.n_estimators,
            save_path=save_path,
            config_path=args.config,
        )
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
