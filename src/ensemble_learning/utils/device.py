"""Device management and array backend (NumPy / CuPy A100 GPU) utilities."""

import logging
from types import ModuleType
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _get_cupy_module() -> Any:
    """Import and return the cupy module if installed."""
    import cupy  # type: ignore[import-not-found]
    return cupy


def _check_cupy_available() -> bool:
    """Check if CuPy is installed and at least one CUDA device is accessible."""
    try:
        cp = _get_cupy_module()
        if hasattr(cp, "cuda"):
            if hasattr(cp.cuda, "is_available") and not cp.cuda.is_available():
                return False
            if hasattr(cp.cuda, "runtime") and hasattr(cp.cuda.runtime, "getDeviceCount"):
                if cp.cuda.runtime.getDeviceCount() <= 0:
                    return False
        return True
    except Exception as exc:
        logger.debug("CuPy/CUDA check failed: %s", exc)
        return False


def get_array_module(device: str = "auto") -> Tuple[ModuleType, str]:
    """Retrieve the array backend module (numpy or cupy) and active device name.

    Parameters
    ----------
    device : str, default="auto"
        Target device: "auto", "cuda", or "cpu".

    Returns
    -------
    tuple[module, str]
        (array_module, device_name) where device_name is "cuda" or "cpu".
    """
    device_lower = str(device).lower()

    if device_lower == "cpu":
        return np, "cpu"

    if device_lower in ("auto", "cuda"):
        if _check_cupy_available():
            try:
                cp = _get_cupy_module()
                return cp, "cuda"
            except Exception as exc:
                logger.warning("Failed to initialize CuPy backend despite check: %s. Falling back to CPU.", exc)
                return np, "cpu"

        if device_lower == "cuda":
            logger.info("CUDA device requested but CuPy/CUDA is unavailable. Falling back to CPU.")
        return np, "cpu"

    raise ValueError(f"Unsupported device: {device}. Supported options are 'auto', 'cuda', 'cpu'.")


def get_device_info(device: str = "auto") -> dict[str, Any]:
    """Inspect and return information about current compute hardware and array backend.

    Parameters
    ----------
    device : str, default="auto"
        Target device: "auto", "cuda", or "cpu".

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "device_name": str ("cpu" or "cuda")
        - "backend_module": str ("numpy" or "cupy")
        - "cuda_available": bool
        - "device_count": int
    """
    cuda_available = _check_cupy_available()
    xp, dev = get_array_module(device)
    count = 0
    if cuda_available:
        try:
            cp = _get_cupy_module()
            count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            count = 1
    return {
        "device_name": dev,
        "backend_module": getattr(xp, "__name__", "numpy"),
        "cuda_available": cuda_available,
        "device_count": count,
    }


def to_numpy(arr: Any) -> np.ndarray:
    """Convert an array-like object (NumPy, CuPy, Pandas, or Python list) to a NumPy ndarray.

    Parameters
    ----------
    arr : Any
        Input array or iterable.

    Returns
    -------
    np.ndarray
        Standard NumPy ndarray.
    """
    if isinstance(arr, np.ndarray):
        return arr

    # CuPy ndarray implements .get() to transfer device array to host NumPy array
    if not isinstance(arr, dict) and hasattr(arr, "get") and callable(arr.get):
        try:
            return np.asarray(arr.get())
        except Exception:
            pass

    # Pandas Series or DataFrame
    if hasattr(arr, "to_numpy") and callable(arr.to_numpy):
        return arr.to_numpy()

    return np.asarray(arr)
