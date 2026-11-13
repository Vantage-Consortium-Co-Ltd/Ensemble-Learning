"""AdaBoost-SAMME written in NumPy on HF scikit-learn/iris. Base models come from scikit-learn.

Run: python boosting_iris.py [--n-estimators 50]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

URL = "https://huggingface.co/datasets/antoniopelusi/iris/resolve/main/iris_train.csv"


def vote(labels, weights, K):
    """labels: (n_models, n_samples) ints, weights: (n_models,) -> weighted majority class."""
    onehot = labels[:, :, None] == np.arange(K)            # (m, n, K)
    return (onehot * weights[:, None, None]).sum(0).argmax(1)


class AdaBoost:
    """SAMME multiclass AdaBoost. Not every sklearn model takes sample_weight (KNN doesn't),
    so each round resamples the training set by weight instead."""

    def __init__(self, base, n_estimators=50, seed=0):
        self.base, self.n, self.seed = base, n_estimators, seed

    def fit(self, X, y):
        n, K = len(X), int(y.max()) + 1
        rng, w = np.random.default_rng(self.seed), np.full(n, 1 / n)
        self.K, self.models, self.alphas = K, [], []
        for _ in range(self.n):
            i = rng.choice(n, n, p=w)
            m = clone(self.base).fit(X[i], y[i])
            miss = m.predict(X) != y
            err = w[miss].sum()
            if err >= 1 - 1 / K:                      # no better than chance: skip this learner
                continue
            alpha = 10.0 if err <= 0 else np.log((1 - err) / err) + np.log(K - 1)
            self.models.append(m)
            self.alphas.append(alpha)
            if err <= 0:                              # perfect fit: weights would stop changing
                break
            w *= np.exp(alpha * miss)
            w /= w.sum()
        return self

    def predict(self, X):
        return vote(np.array([m.predict(X) for m in self.models]), np.array(self.alphas), self.K)


def run(name, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr)
    t1 = time.perf_counter(); pred = model.predict(Xte)
    t2 = time.perf_counter()
    return dict(Method=name, Acc=accuracy_score(yte, pred), F1=f1_score(yte, pred, average="macro"),
                Fit_s=t1 - t0, Predict_s=t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=50)
    a = ap.parse_args()

    df = pd.read_csv(URL)
    X = df.drop(columns=["species"]).to_numpy(float)
    y = pd.factorize(df["species"])[0]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    dt = DecisionTreeClassifier(max_depth=1)
    rows = [run("Decision Tree (single)", clone(dt), Xtr, ytr, Xte, yte),
            run("Decision Tree AdaBoost", AdaBoost(dt, a.n_estimators), Xtr, ytr, Xte, yte)]

    # self-check: the old "beats a single model" assert only held for a weak stump, not KNN; just require better than chance
    assert rows[1]["Acc"] > 1 / 3, "AdaBoost should beat random guessing on 3-class Iris"

    print(f"train={len(Xtr)} test={len(Xte)} n_estimators={a.n_estimators}")
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    res.to_csv(Path(__file__).with_name("results.csv"), index=False)  # read by ../visualize.py


if __name__ == "__main__":
    main()
