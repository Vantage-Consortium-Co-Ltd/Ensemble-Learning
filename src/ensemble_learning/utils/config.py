from pathlib import Path
from typing import Any, Union
import yaml


def load_yaml_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load configuration dictionary from a YAML file.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    return content if content is not None else {}
