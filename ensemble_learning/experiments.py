"""Experiment runners and report formatting."""
import os

import numpy as np

from .bagging import Bagging
from .boosting import AdaBoost
from .classifiers import BASES
from .data import load_iris, standardize, stratified_split
from .metrics import scores, timed

HEADER = (f"{'base':11s} {'method':9s} {'acc':>6s} {'prec':>6s} {'rec':>6s} "
          f"{'f1':>6s} {'fit ms':>9s} {'pred ms':>9s}")


def evaluate(model, X_train, y_train, X_test, y_test, n_classes):
    """Fit + predict one model; return metrics, timings (s) and predictions."""
    _, t_fit = timed(model.fit, X_train, y_train)
    pred, t_pred = timed(model.predict, X_test)
    result = scores(y_test, pred, n_classes)
    result.update(fit=t_fit, pred=t_pred, pred_labels=pred)
    return result


def format_row(name, method, m):
    return (f"{name:11s} {method:9s} {m['acc']:6.3f} {m['prec']:6.3f} {m['rec']:6.3f} "
            f"{m['f1']:6.3f} {m['fit'] * 1e3:9.2f} {m['pred'] * 1e3:9.2f}")


def run_one(job):
    """One (base, seed): Single / Bagging / AdaBoost on the same split.

    Runs inside a worker process, so the ensembles use n_jobs=1 (no oversubscription).
    """
    name, seed, csv, n_est, test_frac = job
    X, y, classes = load_iris(csv)
    tr, te = stratified_split(y, test_frac, np.random.default_rng(seed))
    Xtr, Xte = standardize(X[tr], X[te])
    make, k = BASES[name], len(classes)
    models = {"Single": make(),
              "Bagging": Bagging(make, n_est, n_jobs=1, seed=seed),
              "AdaBoost": AdaBoost(make, n_est, n_jobs=1, seed=seed)}
    out = {}
    for method, model in models.items():
        res = evaluate(model, Xtr, y[tr], Xte, y[te], k)
        res.pop("pred_labels")
        out[method] = res
    return name, out


def report_single_split(X, y, k, args, pool, pct):
    """Table 1: performance + speed for each method on one split (parallel ensembles)."""
    tr, te = stratified_split(y, args.test_frac, np.random.default_rng(42))
    Xtr, Xte = standardize(X[tr], X[te])
    print(f"\n== Single split {pct} (train={len(tr)}, test={len(te)}), parallel ensembles ==")
    print(HEADER)
    for name, make in BASES.items():
        models = (("Single", make()),
                  ("Bagging", Bagging(make, args.n_estimators, args.jobs, 42, pool)),
                  ("AdaBoost", AdaBoost(make, args.n_estimators, 1.0, args.jobs, 42, pool)))
        for method, model in models:
            print(format_row(name, method, evaluate(model, Xtr, y[tr], Xte, y[te], k)))
    return Xtr, Xte, tr, te


def report_speedup(y, k, args, pool, split):
    """Table 2: serial (n_jobs=1) vs parallel timings; asserts identical predictions."""
    Xtr, Xte, tr, te = split
    workers = args.jobs or os.cpu_count()
    print(f"\n== Parallel vs serial speed (n_jobs=1 vs {workers}) ==")
    print(f"{'base':11s} {'method':9s} {'stage':5s} {'serial ms':>10s} {'parallel ms':>12s} {'speedup':>8s}")
    for name, make in BASES.items():
        for method, cls in (("Bagging", Bagging), ("AdaBoost", AdaBoost)):
            kw = {} if method == "Bagging" else {"lr": 1.0}
            ser = evaluate(cls(make, args.n_estimators, n_jobs=1, seed=42, **kw),
                           Xtr, y[tr], Xte, y[te], k)
            par = evaluate(cls(make, args.n_estimators, n_jobs=args.jobs, seed=42,
                               executor=pool, **kw), Xtr, y[tr], Xte, y[te], k)
            assert np.array_equal(ser["pred_labels"], par["pred_labels"])
            for stage in ("fit", "pred"):
                s, p = ser[stage] * 1e3, par[stage] * 1e3
                print(f"{name:11s} {method:9s} {stage:5s} {s:10.2f} {p:12.2f} {s / p:7.2f}x")
    print("(AdaBoost fit is sequential by nature, so only its predict stage can speed up.)")


def report_repeated(args, pool, pct):
    """Table 3: mean metrics over many random splits, experiments run in parallel."""
    jobs = [(n, s, args.csv, args.n_estimators, args.test_frac)
            for n in BASES for s in range(args.seeds)]
    res = list(pool.map(run_one, jobs))
    print(f"\n== Mean over {args.seeds} random {pct} stratified splits (each run single-process) ==")
    print(HEADER + f" {'acc std':>8s}")
    for name in BASES:
        for method in ("Single", "Bagging", "AdaBoost"):
            runs = [o[method] for n, o in res if n == name]
            avg = {key: np.mean([m[key] for m in runs]) for key in ("acc", "prec", "rec", "f1", "fit", "pred")}
            print(format_row(name, method, avg) + f" {np.std([m['acc'] for m in runs]):8.3f}")
