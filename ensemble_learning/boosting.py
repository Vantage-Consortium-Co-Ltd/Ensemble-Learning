"""AdaBoost (multi-class SAMME)."""
import inspect

import numpy as np

from .parallel import parallel_map
from .voting import predict_one, weighted_vote


class AdaBoost:
    """Standard AdaBoost: each round trains on the full data with the current sample
    weights (``fit(X, y, sample_weight=w)``). Classifiers whose fit() has no
    sample_weight (e.g. KNN) fall back to weighted resampling, so ANY classifier works.
    Rounds are sequential by nature, so only prediction is parallel."""

    def __init__(self, make_estimator, n_estimators=25, lr=1.0, n_jobs=None, seed=0, executor=None):
        self.make, self.n_estimators, self.lr = make_estimator, n_estimators, lr
        self.n_jobs, self.seed, self.executor = n_jobs, seed, executor

    def fit(self, X, y):
        n, K = len(y), int(y.max()) + 1
        self.n_classes_ = K
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n)
        self.uses_sample_weight_ = "sample_weight" in inspect.signature(self.make().fit).parameters
        self.estimators_, alphas = [], []
        for _ in range(self.n_estimators):
            if self.uses_sample_weight_:
                est = self.make().fit(X, y, sample_weight=w * n)
            else:
                idx = rng.choice(n, n, replace=True, p=w)
                est = self.make().fit(X[idx], y[idx])
            miss = est.predict(X) != y
            err = np.clip(w[miss].sum(), 0, 1)
            if err >= 1 - 1.0 / K:               # no better than chance: reset weights
                w = np.full(n, 1.0 / n)
                continue
            alpha = self.lr * (np.log((1 - err) / max(err, 1e-10)) + np.log(K - 1))
            self.estimators_.append(est)
            alphas.append(alpha)
            if err <= 1e-10:                     # perfect learner, stop
                break
            w = w * np.exp(alpha * miss)         # up-weight mistakes
            w /= w.sum()
        self.alphas_ = np.array(alphas)
        return self

    def predict(self, X):
        preds = np.array(parallel_map(predict_one, [(e, X) for e in self.estimators_],
                                      self.n_jobs, self.executor))
        return weighted_vote(preds, self.alphas_, self.n_classes_)


# ------------------------------------------------------------ gradient boosting
def _fit_class_tree(args):
    """Fit one regression tree to the residuals of one class, then set each
    leaf value with Friedman's Newton step. Returns (tree, leaf_values)."""
    make, X, resid, p, K = args
    tree = make().fit(X, resid)
    leaf = tree.apply(X)
    size = tree.tree_.node_count
    num = np.bincount(leaf, weights=resid, minlength=size)
    den = np.bincount(leaf, weights=p * (1 - p), minlength=size)
    return tree, (K - 1) / K * num / np.maximum(den, 1e-12)


def _softmax(F):
    e = np.exp(F - F.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class GradientBoosting:
    """Multi-class gradient boosting with softmax log-loss (the 'standard' boosting).

    Each round, for every class k, a small regression tree is fitted to the negative
    gradient (Y_k - p_k) and added to the class score F_k with a learning rate.
    Fully sequential: round m needs the scores after round m-1, and everything runs
    in the calling process (no worker pool).
    """

    def __init__(self, make_regressor, n_estimators=25, lr=0.3):
        self.make, self.n_estimators, self.lr = make_regressor, n_estimators, lr

    def fit(self, X, y):
        n, K = len(y), int(y.max()) + 1
        self.n_classes_ = K
        Y = np.eye(K)[y]
        self.init_ = np.log(np.clip(Y.mean(0), 1e-12, 1))
        F = np.tile(self.init_, (n, 1))
        self.trees_ = []                                   # list of rounds, each K (tree, leaf_values)
        for _ in range(self.n_estimators):
            P = _softmax(F)
            round_trees = [_fit_class_tree((self.make, X, Y[:, k] - P[:, k], P[:, k], K))
                           for k in range(K)]
            for k, (tree, vals) in enumerate(round_trees):
                F[:, k] += self.lr * vals[tree.apply(X)]
            self.trees_.append(round_trees)
        return self

    def predict(self, X):
        F = np.tile(self.init_, (len(X), 1))
        for round_trees in self.trees_:
            for k, (tree, vals) in enumerate(round_trees):
                F[:, k] += self.lr * vals[tree.apply(X)]
        return F.argmax(1)
