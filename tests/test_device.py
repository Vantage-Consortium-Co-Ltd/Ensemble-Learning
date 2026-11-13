import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from ensemble_learning.utils.device import get_array_module, to_numpy
from ensemble_learning.models.base import BaseClassifier


def test_device_fallback_cpu():
    xp, dev_name = get_array_module(device="cpu")
    assert xp.__name__ == "numpy"
    assert dev_name == "cpu"
    arr = xp.array([1, 2, 3])
    np_arr = to_numpy(arr)
    assert isinstance(np_arr, np.ndarray)
    assert np.array_equal(np_arr, np.array([1, 2, 3]))


def test_device_auto_fallback_when_cupy_unavailable():
    xp, dev_name = get_array_module(device="auto")
    # In standard environment without cupy, fallback to cpu numpy
    assert xp.__name__ in ("numpy", "cupy")
    assert dev_name in ("cpu", "cuda")
    if dev_name == "cpu":
        assert xp.__name__ == "numpy"


def test_device_cuda_fallback_when_cupy_unavailable():
    # If cupy is not installed, requesting cuda should safely fallback to cpu
    with patch.dict("sys.modules", {"cupy": None}):
        xp, dev_name = get_array_module(device="cuda")
        assert xp.__name__ == "numpy"
        assert dev_name == "cpu"


def test_device_mock_cupy_success():
    # Simulate an environment where cupy is installed and has cuda device
    mock_cupy = MagicMock()
    mock_cupy.__name__ = "cupy"
    mock_cupy.cuda.is_available.return_value = True
    mock_cupy.cuda.runtime.getDeviceCount.return_value = 1

    with patch.dict("sys.modules", {"cupy": mock_cupy}):
        with patch("ensemble_learning.utils.device._check_cupy_available", return_value=True):
            with patch("ensemble_learning.utils.device._get_cupy_module", return_value=mock_cupy):
                xp, dev_name = get_array_module(device="cuda")
                assert dev_name == "cuda"
                assert xp == mock_cupy

                xp_auto, dev_name_auto = get_array_module(device="auto")
                assert dev_name_auto == "cuda"
                assert xp_auto == mock_cupy


def test_to_numpy_with_cupy_like_object():
    class MockCuPyArray:
        def __init__(self, data):
            self.data = np.array(data)

        def get(self):
            return self.data

    mock_arr = MockCuPyArray([10, 20, 30])
    np_arr = to_numpy(mock_arr)
    assert isinstance(np_arr, np.ndarray)
    assert np.array_equal(np_arr, np.array([10, 20, 30]))


def test_to_numpy_with_python_list_and_numpy():
    arr_list = [1, 2, 3]
    res_list = to_numpy(arr_list)
    assert isinstance(res_list, np.ndarray)
    assert np.array_equal(res_list, np.array([1, 2, 3]))

    arr_np = np.array([4, 5, 6])
    res_np = to_numpy(arr_np)
    assert isinstance(res_np, np.ndarray)
    assert np.array_equal(res_np, arr_np)


def test_base_classifier_abstract():
    # Attempting to instantiate BaseClassifier directly must raise TypeError
    with pytest.raises(TypeError):
        BaseClassifier()


def test_base_classifier_concrete_subclass():
    class DummyClassifier(BaseClassifier):
        def __init__(self):
            self.fitted_ = False

        def fit(self, X: np.ndarray, y: np.ndarray):
            self.fitted_ = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.ones(len(X), dtype=int)

    clf = DummyClassifier()
    X = np.zeros((4, 2))
    y = np.array([1, 1, 0, 1])
    clf.fit(X, y)
    assert clf.fitted_ is True

    preds = clf.predict(X)
    assert np.array_equal(preds, np.array([1, 1, 1, 1]))

    # Score: 3 matches out of 4 -> 0.75
    acc = clf.score(X, y)
    assert np.isclose(acc, 0.75)
