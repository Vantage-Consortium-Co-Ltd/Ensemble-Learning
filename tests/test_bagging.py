import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from ensemble_learning.models.bagging import BaggingClassifier


def test_bagging_bootstrap_sampling():
    model = BaggingClassifier(n_estimators=5, max_samples=0.8, random_state=42)
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model.fit(X, y)
    assert len(model.estimators_) == 5
    for est in model.estimators_:
        assert isinstance(est, DecisionTreeClassifier)


def test_bagging_majority_voting_and_tie_breaking():
    model = BaggingClassifier(n_estimators=4, random_state=42)
    # Manually test internal voting logic on a known prediction matrix
    # Predictions of 4 models on 3 test samples:
    preds = np.array([
        [0, 0, 1, 0],  # majority 0
        [1, 1, 2, 2],  # tie between 1 and 2 -> deterministic tie break
        [2, 2, 2, 2],  # unanimous 2
    ])
    voted = model._majority_vote(preds)
    assert voted[0] == 0
    assert voted[1] in [1, 2]
    assert voted[2] == 2

    # Determinism check across multiple calls
    voted_repeat = model._majority_vote(preds)
    np.testing.assert_array_equal(voted, voted_repeat)


def test_bagging_majority_voting_arbitrary_labels():
    model = BaggingClassifier(n_estimators=3, random_state=42)
    # Non-contiguous or negative class labels
    preds = np.array([
        [-1, -1, 1],   # majority -1
        [10, 20, 20],  # majority 20
    ])
    voted = model._majority_vote(preds)
    assert voted[0] == -1
    assert voted[1] == 20


def test_bagging_single_sample_inference():
    model = BaggingClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(50, 4)
    y = np.random.choice([0, 1, 2], size=50)
    model.fit(X, y)
    # 1D single sample of shape (4,)
    pred_1d = model.predict(X[0])
    assert pred_1d.shape == (1,)
    # 2D single sample of shape (1, 4)
    pred_2d = model.predict(X[0:1])
    assert pred_2d.shape == (1,)
    assert pred_1d[0] == pred_2d[0]


def test_bagging_iris_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    clf = BaggingClassifier(n_estimators=30, max_samples=0.8, n_jobs=2, random_state=42)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = np.mean(preds == y_te)
    assert acc >= 0.90
    assert clf.score(X_te, y_te) == acc


def test_bagging_sequential_execution():
    from sklearn.datasets import load_iris

    data = load_iris()
    clf = BaggingClassifier(n_estimators=5, max_samples=0.8, n_jobs=1, random_state=42)
    clf.fit(data.data, data.target)
    assert len(clf.estimators_) == 5
    preds = clf.predict(data.data[:5])
    assert preds.shape == (5,)


def test_bagging_custom_base_estimator_and_pasting():
    model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=5,
        max_samples=0.5,
        bootstrap=False,
        n_jobs=1,
        random_state=42,
    )
    X = np.random.randn(20, 3)
    y = np.random.choice([0, 1], size=20)
    model.fit(X, y)
    assert len(model.estimators_) == 5
    for est in model.estimators_:
        assert est.max_depth == 2
    preds = model.predict(X)
    assert preds.shape == (20,)
