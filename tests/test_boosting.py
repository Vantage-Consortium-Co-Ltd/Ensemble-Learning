import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ensemble_learning.models.boosting import AdaBoostClassifier, BoostingClassifier


def test_boosting_alias():
    assert AdaBoostClassifier is BoostingClassifier


def test_boosting_initialization_and_weights():
    clf = BoostingClassifier(n_estimators=5, random_state=42)
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1, 2], size=30)
    clf.fit(X, y)
    assert len(clf.estimators_) <= 5
    assert len(clf.estimator_weights_) == len(clf.estimators_)
    assert np.all(clf.estimator_weights_ > 0)


def test_boosting_perfect_weak_learner_early_stop():
    # If weak learner is 100% accurate, error is 0. Should handle zero error without dividing by zero.
    class PerfectLearner:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.random.randn(20, 2)
    y = np.zeros(20, dtype=int)
    clf = BoostingClassifier(base_estimator=PerfectLearner(), n_estimators=10)
    clf.fit(X, y)
    # Must terminate early cleanly without ZeroDivisionError
    assert len(clf.estimators_) == 1
    assert len(clf.estimator_weights_) == 1
    assert clf.estimator_weights_[0] > 0


def test_boosting_single_sample_inference():
    clf = BoostingClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1, 2], size=30)
    clf.fit(X, y)
    pred_1d = clf.predict(X[0])
    pred_2d = clf.predict(X[0:1])
    assert pred_1d.shape == (1,)
    assert pred_2d.shape == (1,)
    assert pred_1d[0] == pred_2d[0]


def test_boosting_iris_accuracy():
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    clf = BoostingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = np.mean(preds == y_te)
    assert acc >= 0.90
    assert clf.score(X_te, y_te) == acc


def test_boosting_worse_than_random_early_stop():
    class AlwaysWrongLearner:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, X):
            # Inverts true labels: error is 1.0 (>= 1 - 1/K = 0.5)
            return np.ones(len(X), dtype=int)

    X = np.random.randn(20, 2)
    y = np.zeros(20, dtype=int)
    clf = BoostingClassifier(base_estimator=AlwaysWrongLearner(), n_estimators=10)
    clf.fit(X, y)
    # Stops early on first round
    assert len(clf.estimators_) == 1
    assert len(clf.estimator_weights_) == 1
    assert clf.estimator_weights_[0] > 0


def test_boosting_binary_classification():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = BoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_boosting_predict_before_fit():
    clf = BoostingClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict(np.array([[1.0, 2.0]]))


def test_boosting_empty_input():
    clf = BoostingClassifier()
    with pytest.raises(ValueError, match="cannot be empty"):
        clf.fit(np.empty((0, 2)), np.empty((0,)))


def test_boosting_mismatched_xy():
    clf = BoostingClassifier()
    with pytest.raises(ValueError, match="Mismatch"):
        clf.fit(np.zeros((10, 2)), np.zeros(8))


def test_boosting_staged_predict():
    data = load_iris()
    clf = BoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(data.data, data.target)
    stages = list(clf.staged_predict(data.data[:5]))
    assert len(stages) == len(clf.estimators_)
    for stage_pred in stages:
        assert stage_pred.shape == (5,)


def test_boosting_predict_proba():
    data = load_iris()
    clf = BoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(data.data, data.target)
    proba = clf.predict_proba(data.data[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-5)
