"""Base classifier interface for ensemble and base estimators."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseClassifier(ABC):
    """Abstract Base Class for all classifiers in the project."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Fit the classifier according to the given training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training vectors.
        y : np.ndarray of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : BaseClassifier
            The fitted classifier.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted classes.
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        preds = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(preds == y_arr))
