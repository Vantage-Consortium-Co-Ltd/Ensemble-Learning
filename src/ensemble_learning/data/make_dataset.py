"""Data ingestion and stratified dataset preparation for Iris classification."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from ensemble_learning.utils.config import load_yaml_config
from ensemble_learning.utils.logger import setup_logger

logger = setup_logger("ensemble_learning.data.make_dataset")

FEATURE_COLUMNS = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
TARGET_COLUMN = "target"
SPECIES_MAP = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
SKLEARN_RENAME_MAP = {
    "sepal length (cm)": "sepal-length",
    "sepal width (cm)": "sepal-width",
    "petal length (cm)": "petal-length",
    "petal width (cm)": "petal-width",
}


def load_iris_data(
    raw_data_dir: Union[str, Path] = "data/raw",
    force_download: bool = False,
    dataset_name: str = "antoniopelusi/iris",
) -> pd.DataFrame:
    """Load Iris dataset from Hugging Face with local cache and sklearn fallback.

    Parameters
    ----------
    raw_data_dir : Union[str, Path], default="data/raw"
        Directory where raw cached dataset is stored.
    force_download : bool, default=False
        If True, ignore local cache and attempt to download fresh data.
    dataset_name : str, default="antoniopelusi/iris"
        Hugging Face repository ID for the dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with 4 numerical feature columns and 1 integer 'target' column.
    """
    raw_dir_path = Path(raw_data_dir)
    cache_file = raw_dir_path / "iris.csv"

    # 1. Return from local cache if present and not force_download
    if cache_file.is_file() and not force_download:
        logger.info("Loading cached Iris dataset from %s", cache_file)
        df = pd.read_csv(cache_file)
        # Ensure correct column ordering and types
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        for col in FEATURE_COLUMNS:
            df[col] = df[col].astype(float)
        return df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    raw_dir_path.mkdir(parents=True, exist_ok=True)

    # 2. Try loading from Hugging Face Datasets
    try:
        import datasets  # type: ignore[import-untyped]

        logger.info("Fetching Iris dataset from Hugging Face: %s", dataset_name)
        hf_dataset = datasets.load_dataset(dataset_name)
        split_key = "train" if "train" in hf_dataset else list(hf_dataset.keys())[0]
        df_raw = hf_dataset[split_key].to_pandas()

        if "species" in df_raw.columns:
            df_raw[TARGET_COLUMN] = df_raw["species"].map(SPECIES_MAP).astype(int)
        elif TARGET_COLUMN not in df_raw.columns:
            raise KeyError(f"Neither 'species' nor 'target' column found in {dataset_name}")

        df = pd.DataFrame()
        for col in FEATURE_COLUMNS:
            df[col] = df_raw[col].astype(float)
        df[TARGET_COLUMN] = df_raw[TARGET_COLUMN].astype(int)

    except Exception as exc:
        logger.warning(
            "Failed to load dataset from Hugging Face (%s). Falling back to sklearn.datasets.load_iris.",
            exc,
        )
        from sklearn.datasets import load_iris

        sklearn_iris = load_iris(as_frame=True)
        df_sklearn = sklearn_iris.frame.rename(columns=SKLEARN_RENAME_MAP)
        df = pd.DataFrame()
        for col in FEATURE_COLUMNS:
            df[col] = df_sklearn[col].astype(float)
        df[TARGET_COLUMN] = df_sklearn[TARGET_COLUMN].astype(int)

    # 3. Cache to raw data directory
    df.to_csv(cache_file, index=False)
    logger.info("Cached Iris dataset to %s (shape: %s)", cache_file, df.shape)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
    processed_dir: Optional[Union[str, Path]] = "data/processed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified split of Iris dataset into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing feature columns and 'target'.
    train_ratio : float, default=0.8
        Proportion of samples per class assigned to training set.
    random_state : int, default=42
        Seed for reproducibility of stratified permutations.
    processed_dir : Optional[Union[str, Path]], default="data/processed"
        Directory where processed train.csv and test.csv are saved.
        If None, files are not saved to disk.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_test, y_train, y_test)
        - X_train: np.ndarray of shape (n_train, n_features), float64
        - X_test: np.ndarray of shape (n_test, n_features), float64
        - y_train: np.ndarray of shape (n_train,), int64
        - y_test: np.ndarray of shape (n_test,), int64
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    targets = df[TARGET_COLUMN].to_numpy()
    unique_classes = np.unique(targets)

    rng = np.random.RandomState(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for cls in unique_classes:
        cls_idx = np.where(targets == cls)[0]
        permuted = rng.permutation(cls_idx)
        n_train = int(np.floor(train_ratio * len(cls_idx)))
        train_indices.extend(permuted[:n_train].tolist())
        test_indices.extend(permuted[n_train:].tolist())

    # Shuffle combined train and test indices reproducibly
    shuffled_train_idx = rng.permutation(np.array(train_indices, dtype=np.int64))
    shuffled_test_idx = rng.permutation(np.array(test_indices, dtype=np.int64))

    X_train = df.iloc[shuffled_train_idx][feature_cols].to_numpy(dtype=np.float64)
    y_train = df.iloc[shuffled_train_idx][TARGET_COLUMN].to_numpy(dtype=np.int64)
    X_test = df.iloc[shuffled_test_idx][feature_cols].to_numpy(dtype=np.float64)
    y_test = df.iloc[shuffled_test_idx][TARGET_COLUMN].to_numpy(dtype=np.int64)

    if processed_dir is not None:
        proc_path = Path(processed_dir)
        proc_path.mkdir(parents=True, exist_ok=True)
        train_df = df.iloc[shuffled_train_idx].copy()
        test_df = df.iloc[shuffled_test_idx].copy()
        train_df.to_csv(proc_path / "train.csv", index=False)
        test_df.to_csv(proc_path / "test.csv", index=False)
        logger.info(
            "Saved processed splits to %s: train=%s, test=%s",
            proc_path,
            train_df.shape,
            test_df.shape,
        )

    return X_train, X_test, y_train, y_test


def load_dataset_from_config(
    config_path: Union[str, Path] = "configs/train_config.yaml",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split dataset according to YAML configuration.

    Parameters
    ----------
    config_path : Union[str, Path], default="configs/train_config.yaml"
        Path to training configuration YAML file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_test, y_train, y_test)
    """
    config = load_yaml_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    random_state = int(data_cfg.get("random_state", 42))
    dataset_name = data_cfg.get("dataset_name", "antoniopelusi/iris")

    df = load_iris_data(
        raw_data_dir=raw_dir,
        dataset_name=dataset_name,
    )
    return split_train_test(
        df=df,
        train_ratio=train_ratio,
        random_state=random_state,
        processed_dir=processed_dir,
    )
