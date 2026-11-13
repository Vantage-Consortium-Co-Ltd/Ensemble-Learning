"""Weighted majority voting shared by Bagging and AdaBoost."""
import numpy as np


def weighted_vote(preds, weights, n_classes):
    """preds: (n_estimators, n_samples) int labels -> (n_samples,) winning labels."""
    n_samples = preds.shape[1]
    votes = np.zeros((n_samples, n_classes))
    cols = np.arange(n_samples)
    for est_preds, w in zip(preds, weights):
        np.add.at(votes, (cols, est_preds), w)
    return votes.argmax(1)


def predict_one(args):
    """Worker: (fitted_estimator, X) -> labels. Top-level so it can be pickled."""
    est, X = args
    return est.predict(X)
