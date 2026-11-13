"""Bagging (parallel, joblib) on HF scikit-learn/iris. Base models come from scikit-learn.

Run: python bagging_iris.py [--n-estimators 50] [--scale 1] [--backend threads|processes]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

URL = "https://huggingface.co/datasets/antoniopelusi/iris/resolve/main/iris_train.csv"


def vote(labels, K):
    """labels: (n_models, n_samples) ints -> majority class per sample."""
    return (labels[:, :, None] == np.arange(K)).sum(0).argmax(1)


def _fit_bootstrap(base, X, y, seed):
    i = np.random.default_rng(seed).integers(0, len(X), len(X))
    return clone(base).fit(X[i], y[i])


class Bagging:
    def __init__(self, base, n_estimators=50, n_jobs=-1, backend="threads", seed=99):
        self.base, self.n, self.n_jobs, self.backend, self.seed = base, n_estimators, n_jobs, backend, seed

    def fit(self, X, y):
        self.K = int(y.max()) + 1
        self.models = Parallel(n_jobs=self.n_jobs, prefer=self.backend)(
            delayed(_fit_bootstrap)(self.base, X, y, self.seed + i) for i in range(self.n))
        return self

    def predict(self, X):
        return vote(np.array([m.predict(X) for m in self.models]), self.K)


def run(name, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr)
    t1 = time.perf_counter(); pred = model.predict(Xte)
    t2 = time.perf_counter()
    return pred, dict(Method=name, Acc=accuracy_score(yte, pred), F1=f1_score(yte, pred, average="macro"),
                      Fit_s=t1 - t0, Predict_s=t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--scale", type=int, default=1, help="tile the TRAIN set k times to expose parallel speedup")
    # threads: sklearn fits release the GIL and there is no worker spawn cost (processes took ~12s to start on Windows)
    ap.add_argument("--backend", default="threads", choices=["threads", "processes"])
    a = ap.parse_args()

    df = pd.read_csv(URL)
    X = df.drop(columns=["species"]).to_numpy(float)
    y = pd.factorize(df["species"])[0]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, ytr = np.tile(Xtr, (a.scale, 1)), np.tile(ytr, a.scale)

    models = {"Decision Tree": DecisionTreeClassifier(max_depth=1)}
    Parallel(n_jobs=-1, prefer=a.backend)(delayed(abs)(i) for i in range(64))  # warm the worker pool

    rows = []
    for k, m in models.items():
        rows.append(run(f"{k} (single)", clone(m), Xtr, ytr, Xte, yte)[1])
        rows.append(run(f"{k} Bagging parallel", Bagging(m, a.n_estimators, -1, a.backend), Xtr, ytr, Xte, yte)[1])

    print(f"train={len(Xtr)} test={len(Xte)} n_estimators={a.n_estimators} backend={a.backend}")
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    res.to_csv(Path(__file__).with_name("results.csv"), index=False)  # read by ../visualize.py


if __name__ == "__main__":
    main()
