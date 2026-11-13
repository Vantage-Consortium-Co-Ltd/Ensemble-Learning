"""Bagging: bootstrap-aggregating with majority vote."""
import numpy as np

from .parallel import parallel_map
from .voting import predict_one, weighted_vote


def _fit_on_bootstrap(args):
    """Worker: fit a fresh estimator on one bootstrap sample (top-level for pickling)."""
    make, X, y, seed = args
    idx = np.random.default_rng(seed).integers(0, len(y), len(y))   # sample with replacement
    return make().fit(X[idx], y[idx])


class Bagging:
    """Every estimator is independent, so both fit and predict run in parallel."""

    def __init__(self, make_estimator, n_estimators=25, n_jobs=None, seed=0, executor=None):
        self.make, self.n_estimators = make_estimator, n_estimators
        self.n_jobs, self.seed, self.executor = n_jobs, seed, executor

    def fit(self, X, y):
        self.n_classes_ = int(y.max()) + 1
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_estimators)
        jobs = [(self.make, X, y, int(s)) for s in seeds]
        self.estimators_ = parallel_map(_fit_on_bootstrap, jobs, self.n_jobs, self.executor)
        return self

    def predict(self, X):
        preds = np.array(parallel_map(predict_one, [(e, X) for e in self.estimators_],
                                      self.n_jobs, self.executor))
        return weighted_vote(preds, np.ones(len(preds)), self.n_classes_)
