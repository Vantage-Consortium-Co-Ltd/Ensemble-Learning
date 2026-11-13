"""AdaBoost (multi-class SAMME)."""
import numpy as np

from .parallel import parallel_map
from .voting import predict_one, weighted_vote


class AdaBoost:
    """Sample weights are honoured by weighted resampling, so ANY classifier works
    (KNN / Naive Bayes have no sample_weight). Rounds are sequential by nature, so
    only prediction over the fitted estimators is parallel."""

    def __init__(self, make_estimator, n_estimators=25, lr=1.0, n_jobs=None, seed=0, executor=None):
        self.make, self.n_estimators, self.lr = make_estimator, n_estimators, lr
        self.n_jobs, self.seed, self.executor = n_jobs, seed, executor

    def fit(self, X, y):
        n, K = len(y), int(y.max()) + 1
        self.n_classes_ = K
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n)
        self.estimators_, alphas = [], []
        for _ in range(self.n_estimators):
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
