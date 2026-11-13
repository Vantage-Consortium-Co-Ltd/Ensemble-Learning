"""Pure NumPy Multi-class AdaBoost (SAMME) Classifier."""

import copy
from typing import Any, Generator, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from ensemble_learning.models.base import BaseClassifier


class AdaBoostClassifier(BaseClassifier):
    """Multi-class AdaBoost classifier implementing SAMME in pure NumPy.

    Parameters
    ----------
    base_estimator : Optional[Any], default=None
        The base estimator from which the boosted ensemble is built.
        If None, defaults to DecisionTreeClassifier(max_depth=1).
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit or worse-than-random fit, learning terminates early.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    random_state : Optional[int], default=42
        Controls the randomness of the base estimators.

    Attributes
    ----------
    estimators_ : List[Any]
        The collection of fitted base estimators.
    estimator_weights_ : np.ndarray
        Weights (alpha) for each estimator in the boosted ensemble.
    estimator_errors_ : np.ndarray
        Classification error for each estimator in the boosted ensemble.
    classes_ : np.ndarray
        Unique class labels identified during fit.
    n_classes_ : int
        Number of unique classes.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List[Any] = []
        self.estimator_weights_: np.ndarray = np.array([], dtype=float)
        self.estimator_errors_: np.ndarray = np.array([], dtype=float)
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the AdaBoost SAMME ensemble classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training vectors.
        y : np.ndarray of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : AdaBoostClassifier
            The fitted classifier.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        n_samples = X_arr.shape[0]
        if n_samples == 0 or y_arr.shape[0] == 0:
            raise ValueError("Training dataset cannot be empty.")
        if n_samples != y_arr.shape[0]:
            raise ValueError(
                f"Mismatch between X samples ({n_samples}) and y samples ({y_arr.shape[0]})."
            )

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        K = self.n_classes_

        rng = (
            np.random.RandomState(self.random_state)
            if self.random_state is not None
            else None
        )

        template = (
            self.base_estimator
            if self.base_estimator is not None
            else DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
        )

        # Initialize uniform sample weights
        w = np.ones(n_samples, dtype=float) / float(n_samples)

        self.estimators_ = []
        estimator_weights: List[float] = []
        estimator_errors: List[float] = []

        for m in range(self.n_estimators):
            # Clone base estimator
            try:
                est = clone(template)
            except Exception:
                est = copy.deepcopy(template)

            # Assign seed to base estimator if applicable and unset
            if hasattr(est, "random_state") and est.random_state is None and rng is not None:
                est.random_state = rng.randint(0, np.iinfo(np.int32).max)

            # Fit estimator with sample_weight
            try:
                est.fit(X_arr, y_arr, sample_weight=w)
            except TypeError:
                est.fit(X_arr, y_arr)

            y_pred = est.predict(X_arr)
            incorrect = (y_arr != y_pred)

            # Weighted classification error
            w_sum = np.sum(w)
            err_m = float(np.sum(w * incorrect) / w_sum)

            # Zero-error catch: perfect learner
            if err_m <= 0.0:
                self.estimators_.append(est)
                estimator_weights.append(1.0)
                estimator_errors.append(0.0)
                break

            # Worse-than-random catch: random guessing error is 1 - 1/K
            if err_m >= 1.0 - (1.0 / float(K)):
                if len(self.estimators_) == 0:
                    self.estimators_.append(est)
                    estimator_weights.append(1.0)
                    estimator_errors.append(err_m)
                break

            # SAMME alpha computation
            alpha_m = float(
                self.learning_rate
                * (
                    np.log((1.0 - err_m) / err_m)
                    + (np.log(float(K - 1)) if K > 1 else 0.0)
                )
            )

            self.estimators_.append(est)
            estimator_weights.append(alpha_m)
            estimator_errors.append(err_m)

            # Update and normalize sample weights
            if m < self.n_estimators - 1:
                w *= np.exp(alpha_m * incorrect)
                new_sum = np.sum(w)
                if new_sum > 0 and np.isfinite(new_sum):
                    w /= new_sum
                else:
                    w = np.ones(n_samples, dtype=float) / float(n_samples)

        self.estimator_weights_ = np.asarray(estimator_weights, dtype=float)
        self.estimator_errors_ = np.asarray(estimator_errors, dtype=float)
        return self

    def _check_is_fitted(self) -> None:
        """Verify that the ensemble has been fitted."""
        if not hasattr(self, "estimators_") or len(self.estimators_) == 0:
            raise ValueError(
                "This AdaBoostClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the SAMME decision score matrix for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input test samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples, n_classes)
            Weighted vote scores for each class.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = np.atleast_2d(X_arr)
        elif X_arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D array.")

        n_samples = X_arr.shape[0]
        K = self.n_classes_
        scores = np.zeros((n_samples, K), dtype=float)

        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = est.predict(X_arr)
            for k, cls_label in enumerate(self.classes_):
                scores[:, k] += alpha * (y_pred == cls_label)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted classes.
        """
        scores = self.decision_function(X)
        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Estimated class probabilities.
        """
        scores = self.decision_function(X)
        sum_scores = np.sum(scores, axis=1, keepdims=True)
        # Safe normalization
        with np.errstate(divide="ignore", invalid="ignore"):
            proba = np.where(
                sum_scores > 0,
                scores / sum_scores,
                1.0 / float(self.n_classes_),
            )
        return proba

    def staged_predict(self, X: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Yield predictions at each boosting iteration.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input samples.

        Yields
        ------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted classes after each boosting iteration.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = np.atleast_2d(X_arr)
        elif X_arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D array.")

        n_samples = X_arr.shape[0]
        K = self.n_classes_
        scores = np.zeros((n_samples, K), dtype=float)

        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = est.predict(X_arr)
            for k, cls_label in enumerate(self.classes_):
                scores[:, k] += alpha * (y_pred == cls_label)
            best_indices = np.argmax(scores, axis=1)
            yield self.classes_[best_indices]
