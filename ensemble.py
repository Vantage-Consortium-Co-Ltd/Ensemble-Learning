"""Bagging and Boosting (AdaBoost-SAMME) implemented with NumPy only.

* The ensemble logic (bootstrap sampling, weight updates, voting, metrics,
  data loading, scaling, splitting) is pure NumPy.
* Base classifiers (KNN, GaussianNB, DecisionTree, ...) come from scikit-learn.
* Parallelism: concurrent.futures.ProcessPoolExecutor.
    - Bagging: every estimator is independent -> fit AND predict in parallel.
    - Boosting: rounds are inherently sequential, so each round's fit is
      single-process, but prediction over all estimators is parallel.
      Independent experiments (model x seed) are also run in parallel.

Dataset: Kaggle iris-dataset (vikrishnan), file iris.data.csv next to this file
(or pass --csv).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- data (numpy)
def load_iris(path):
    """Reads iris.data.csv: 4 numeric features + species label per row.
    Tolerates a header row, an Id column and blank lines."""
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]                          # header row
    data = np.array(rows, dtype=str)
    X = data[:, -5:-1].astype(float)             # 4 features (skips Id if present)
    classes, y = np.unique(data[:, -1], return_inverse=True)
    return X, y, classes


def stratified_split(y, test_frac, rng):
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.append(idx[: int(round(len(idx) * test_frac))])
    test = np.concatenate(test)
    train = np.setdiff1d(np.arange(len(y)), test)
    return rng.permutation(train), test


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def confusion_matrix(y, p, k):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)
    return cm


def scores(y, p, k):
    """accuracy + macro precision / recall / F1 (NumPy)."""
    cm = confusion_matrix(y, p, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(acc=tp.sum() / cm.sum(), prec=prec.mean(), rec=rec.mean(), f1=f1.mean())


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t


# --------------------------------------------------------- parallel worker fns
# (top level so they can be pickled by multiprocessing)
def _fit_bag(args):
    make, X, y, seed = args
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), len(y))        # bootstrap sample w/ replacement
    return make().fit(X[idx], y[idx])


def _predict(args):
    est, X = args
    return est.predict(X)


def _vote(preds, weights, n_classes):
    """Weighted vote. preds: (n_est, n_samples) ints -> (n_samples,) labels."""
    n_est, n = preds.shape
    scores = np.zeros((n, n_classes))
    cols = np.arange(n)
    for m in range(n_est):
        np.add.at(scores, (cols, preds[m]), weights[m])
    return scores.argmax(1), scores


def _map(fn, items, n_jobs, executor):
    """Serial in-process if n_jobs==1; shared executor if given; else a temp pool."""
    if executor is not None:
        return list(executor.map(fn, items))
    if n_jobs == 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(fn, items))


def _noop(_):
    return 0


# ------------------------------------------------------------------- Bagging
class Bagging:
    def __init__(self, make_estimator, n_estimators=25, n_jobs=None, seed=0, executor=None):
        self.make, self.n_estimators = make_estimator, n_estimators
        self.n_jobs, self.seed, self.executor = n_jobs, seed, executor

    def fit(self, X, y):
        self.n_classes_ = int(y.max()) + 1
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_estimators)
        self.estimators_ = _map(_fit_bag, [(self.make, X, y, int(s)) for s in seeds],
                                self.n_jobs, self.executor)
        return self

    def predict(self, X):
        preds = np.array(_map(_predict, [(e, X) for e in self.estimators_],
                              self.n_jobs, self.executor))
        return _vote(preds, np.ones(len(preds)), self.n_classes_)[0]


# ------------------------------------------------------------------ Boosting
class AdaBoost:
    """Multi-class AdaBoost (SAMME). Weights are honoured by weighted
    resampling, so ANY classifier works (KNN / NB have no sample_weight)."""

    def __init__(self, make_estimator, n_estimators=25, lr=1.0, n_jobs=None, seed=0, executor=None):
        self.make, self.n_estimators, self.lr = make_estimator, n_estimators, lr
        self.n_jobs, self.seed, self.executor = n_jobs, seed, executor

    def fit(self, X, y):
        n, K = len(y), int(y.max()) + 1
        self.n_classes_ = K
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n)
        self.estimators_, self.alphas_ = [], []
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
            self.alphas_.append(alpha)
            if err <= 1e-10:                     # perfect learner, stop
                break
            w = w * np.exp(alpha * miss)
            w /= w.sum()
        self.alphas_ = np.array(self.alphas_)
        return self

    def predict(self, X):
        preds = np.array(_map(_predict, [(e, X) for e in self.estimators_],
                              self.n_jobs, self.executor))
        return _vote(preds, self.alphas_, self.n_classes_)[0]


# ------------------------------------------------------------ base classifiers
def knn():   return KNeighborsClassifier(n_neighbors=5)
def nb():    return GaussianNB()
def stump(): return DecisionTreeClassifier(max_depth=1)
def tree():  return DecisionTreeClassifier(max_depth=3)

BASES = {"KNN": knn, "NaiveBayes": nb, "Stump": stump, "Tree(d=3)": tree}


# ------------------------------------------------------------------ experiment
def evaluate(model, Xtr, ytr, Xte, yte, k):
    """Fit + predict one model, returning metrics and wall-clock timings."""
    model, t_fit = timed(lambda: model.fit(Xtr, ytr))
    pred, t_pred = timed(model.predict, Xte)
    m = scores(yte, pred, k)
    m.update(fit=t_fit, pred=t_pred, pred_labels=pred)
    return m


def run_one(job):
    """One (base, seed): single / Bagging / AdaBoost on the same split. Runs in
    a worker process, so ensembles inside use n_jobs=1 (no oversubscription)."""
    name, seed, csv, n_est, test_frac = job
    X, y, classes = load_iris(csv)
    tr, te = stratified_split(y, test_frac, np.random.default_rng(seed))
    Xtr, Xte = standardize(X[tr], X[te])
    make, k = BASES[name], len(classes)
    out = {"Single": evaluate(make(), Xtr, y[tr], Xte, y[te], k),
           "Bagging": evaluate(Bagging(make, n_est, 1, seed), Xtr, y[tr], Xte, y[te], k),
           "AdaBoost": evaluate(AdaBoost(make, n_est, 1.0, 1, seed), Xtr, y[tr], Xte, y[te], k)}
    for v in out.values():
        v.pop("pred_labels")
    return name, out


HDR = f"{'base':11s} {'method':9s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'fit ms':>9s} {'pred ms':>9s}"


def row(name, method, m):
    return (f"{name:11s} {method:9s} {m['acc']:6.3f} {m['prec']:6.3f} {m['rec']:6.3f} "
            f"{m['f1']:6.3f} {m['fit'] * 1e3:9.2f} {m['pred'] * 1e3:9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "iris.data.csv"))
    ap.add_argument("--n-estimators", type=int, default=25)
    ap.add_argument("--test-frac", type=float, default=0.2, help="test fraction (0.2 = 80/20 split)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    a = ap.parse_args()
    pct = f"{int(round((1 - a.test_frac) * 100))}/{int(round(a.test_frac * 100))}"

    X, y, classes = load_iris(a.csv)
    k = len(classes)
    print(f"Loaded {a.csv}: {X.shape[0]} samples, {X.shape[1]} features, classes={list(classes)}")
    print(f"CPU cores: {os.cpu_count()}, workers: {a.jobs or os.cpu_count()}, estimators: {a.n_estimators}")

    # One shared pool, warmed up, so timings measure work rather than process startup
    t = time.perf_counter()
    pool = ProcessPoolExecutor(a.jobs)
    list(pool.map(_noop, range((a.jobs or os.cpu_count()) * 4)))
    print(f"Process-pool startup (one-off, excluded from timings below): {(time.perf_counter() - t) * 1e3:.0f} ms")

    # 1) One split: performance + speed of each method, parallel ensembles
    tr, te = stratified_split(y, a.test_frac, np.random.default_rng(42))
    Xtr, Xte = standardize(X[tr], X[te])
    print(f"\n== Single split {pct} (train={len(tr)}, test={len(te)}), parallel ensembles ==")
    print(HDR)
    for name, make in BASES.items():
        for method, model in (("Single", make()),
                              ("Bagging", Bagging(make, a.n_estimators, a.jobs, 42, pool)),
                              ("AdaBoost", AdaBoost(make, a.n_estimators, 1.0, a.jobs, 42, pool))):
            print(row(name, method, evaluate(model, Xtr, y[tr], Xte, y[te], k)))

    # 2) Parallel vs serial speed (same seeds -> identical models, so same accuracy)
    print(f"\n== Parallel vs serial speed (n_jobs=1 vs {a.jobs or os.cpu_count()}) ==")
    print(f"{'base':11s} {'method':9s} {'stage':5s} {'serial ms':>10s} {'parallel ms':>12s} {'speedup':>8s}")
    for name, make in BASES.items():
        for method, cls in (("Bagging", Bagging), ("AdaBoost", AdaBoost)):
            kw = {} if method == "Bagging" else {"lr": 1.0}
            ser = evaluate(cls(make, a.n_estimators, n_jobs=1, seed=42, **kw), Xtr, y[tr], Xte, y[te], k)
            par = evaluate(cls(make, a.n_estimators, n_jobs=a.jobs, seed=42, executor=pool, **kw), Xtr, y[tr], Xte, y[te], k)
            assert np.array_equal(ser.pop("pred_labels"), par.pop("pred_labels"))
            for stage in ("fit", "pred"):
                s_, p_ = ser[stage] * 1e3, par[stage] * 1e3
                print(f"{name:11s} {method:9s} {stage:5s} {s_:10.2f} {p_:12.2f} {s_ / p_:7.2f}x")
    print("(AdaBoost fit is sequential by nature, so only its predict stage can speed up.)")

    # 3) Many random splits, experiments run in parallel; mean of each metric
    res = list(pool.map(run_one, [(n, s, a.csv, a.n_estimators, a.test_frac)
                                    for n in BASES for s in range(a.seeds)]))
    pool.shutdown()
    print(f"\n== Mean over {a.seeds} random {pct} stratified splits (each run single-process) ==")
    print(HDR + f" {'acc std':>8s}")
    for n in BASES:
        for method in ("Single", "Bagging", "AdaBoost"):
            ms = [o[method] for nm, o in res if nm == n]
            avg = {key: np.mean([m[key] for m in ms]) for key in ms[0]}
            print(row(n, method, avg) + f" {np.std([m['acc'] for m in ms]):8.3f}")


if __name__ == "__main__":
    main()
