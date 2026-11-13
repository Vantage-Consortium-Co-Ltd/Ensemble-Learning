"""Pure NumPy Bagging Classifier with parallel execution and voting."""

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from ensemble_learning.models.base import BaseClassifier


def _fit_single_estimator(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
) -> Any:
    """Fit a single base estimator on the sampled subset of data.

    Module-level worker function to ensure compatibility with
    ProcessPoolExecutor across platforms (including Windows spawn).

    Parameters
    ----------
    estimator : Any
        An unfitted estimator clone.
    X : np.ndarray
        Full training feature matrix.
    y : np.ndarray
        Full training target vector.
    sample_indices : np.ndarray
        Indices for bootstrap or subsample training.

    Returns
    -------
    fitted_estimator : Any
        The fitted base estimator.
    """
    X_subset = X[sample_indices]
    y_subset = y[sample_indices]
    estimator.fit(X_subset, y_subset)
    return estimator


class BaggingClassifier(BaseClassifier):
    """Bagging (Bootstrap Aggregating) ensemble classifier using pure NumPy.

    Parameters
    ----------
    base_estimator : Optional[Any], default=None
        The base estimator from which the ensemble is built.
        If None, defaults to DecisionTreeClassifier().
    n_estimators : int, default=50
        Number of base estimators in the ensemble.
    max_samples : Union[int, float], default=0.8
        The number or ratio of samples to draw from X to train each base estimator.
        If float, draws int(max_samples * n_samples) samples.
        If int, draws max_samples samples.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement (bootstrap) or without (pasting).
    n_jobs : Optional[int], default=4
        Number of parallel jobs to run for fitting. If 1 or None, runs sequentially.
        If > 1, uses ProcessPoolExecutor. If -1, uses all available CPUs.
    random_state : Optional[int], default=42
        Controls the randomness of bootstrap sampling and base estimators.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 50,
        max_samples: Union[int, float] = 0.8,
        bootstrap: bool = True,
        n_jobs: Optional[int] = 4,
        random_state: Optional[int] = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators_: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """Fit the ensemble of base estimators on bootstrap samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training vectors.
        y : np.ndarray of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : BaggingClassifier
            Fitted classifier.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        n_samples = X_arr.shape[0]
        if n_samples == 0:
            raise ValueError("Training dataset cannot be empty.")

        # Determine number of samples per estimator
        if isinstance(self.max_samples, (int, np.integer)):
            sample_size = min(int(self.max_samples), n_samples)
        elif isinstance(self.max_samples, float):
            sample_size = max(1, int(self.max_samples * n_samples))
        else:
            sample_size = int(self.max_samples)

        rng = np.random.RandomState(self.random_state)

        # Generate sample indices for each estimator
        sample_indices_list = [
            rng.choice(n_samples, size=sample_size, replace=self.bootstrap)
            for _ in range(self.n_estimators)
        ]

        # Prepare base estimator clones
        template = (
            self.base_estimator
            if self.base_estimator is not None
            else DecisionTreeClassifier()
        )

        cloned_estimators = []
        for _ in range(self.n_estimators):
            try:
                est = clone(template)
            except Exception:
                est = copy.deepcopy(template)

            # Assign seed to base estimator if applicable and unset
            if (
                hasattr(est, "random_state")
                and est.random_state is None
                and self.random_state is not None
            ):
                est.random_state = rng.randint(0, np.iinfo(np.int32).max)

            cloned_estimators.append(est)

        # Resolve effective number of workers
        effective_jobs = self.n_jobs
        if effective_jobs is not None and effective_jobs < 0:
            effective_jobs = os.cpu_count() or 1

        # Train estimators: parallel if effective_jobs > 1, else sequential fallback
        if effective_jobs is not None and effective_jobs > 1:
            with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
                futures = [
                    executor.submit(_fit_single_estimator, est, X_arr, y_arr, indices)
                    for est, indices in zip(cloned_estimators, sample_indices_list)
                ]
                self.estimators_ = [f.result() for f in futures]
        else:
            self.estimators_ = [
                _fit_single_estimator(est, X_arr, y_arr, indices)
                for est, indices in zip(cloned_estimators, sample_indices_list)
            ]

        return self

    def _majority_vote(self, preds: np.ndarray) -> np.ndarray:
        """Compute majority vote across estimators per sample with deterministic tie breaking.

        Parameters
        ----------
        preds : np.ndarray of shape (n_samples, n_estimators) or (n_estimators,)
            Matrix of predicted class labels from each estimator.

        Returns
        -------
        voted : np.ndarray of shape (n_samples,)
            Ensemble majority voted class labels.
        """
        preds_arr = np.asarray(preds)
        if preds_arr.size == 0:
            return np.array([], dtype=int)

        if preds_arr.ndim == 1:
            preds_arr = preds_arr.reshape(1, -1)

        classes, inverse = np.unique(preds_arr, return_inverse=True)
        inv = inverse.reshape(preds_arr.shape)

        # Vectorized class counts: shape (n_samples, n_classes)
        counts = (inv[:, :, None] == np.arange(len(classes))).sum(axis=1)

        # np.argmax returns first occurrence of maximum count -> deterministic tie-break
        best_indices = np.argmax(counts, axis=1)
        return classes[best_indices]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Supports both 2D input (n_samples, n_features) and 1D single-sample input (n_features,).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input test samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,) or (1,)
            Predicted class labels.
        """
        if not self.estimators_:
            raise RuntimeError("BaggingClassifier must be fitted before calling predict.")

        X_2d = np.atleast_2d(X)
        if X_2d.shape[0] == 0:
            return np.array([])

        # Gather predictions from all estimators: list of shape (n_samples,) -> matrix (n_samples, n_estimators)
        all_preds = np.column_stack([est.predict(X_2d) for est in self.estimators_])

        return self._majority_vote(all_preds)
