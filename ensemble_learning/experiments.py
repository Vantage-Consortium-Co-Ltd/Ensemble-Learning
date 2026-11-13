"""Experiment runners and report formatting."""
import os

import numpy as np

from .bagging import Bagging
from .boosting import GradientBoosting
from .classifiers import BASES, GRAD_BASES
from .data import load_iris, standardize, stratified_split
from .metrics import scores, timed

HEADER = (f"{'base':13s} {'method':9s} {'acc':>6s} {'prec':>6s} {'rec':>6s} "
          f"{'f1':>6s} {'fit ms':>9s} {'pred ms':>9s}")


def evaluate(model, X_train, y_train, X_test, y_test, n_classes):
    """Fit + predict one model; return metrics, timings (s) and predictions."""
    _, t_fit = timed(model.fit, X_train, y_train)
    pred, t_pred = timed(model.predict, X_test)
    result = scores(y_test, pred, n_classes)
    result.update(fit=t_fit, pred=t_pred, pred_labels=pred)
    return result


def format_row(name, method, m):
    return (f"{name:13s} {method:9s} {m['acc']:6.3f} {m['prec']:6.3f} {m['rec']:6.3f} "
            f"{m['f1']:6.3f} {m['fit'] * 1e3:9.2f} {m['pred'] * 1e3:9.2f}")


def run_one(job):
    """One (base, seed) on one random split.

    Classifier bases -> Single + Bagging; regression-tree bases -> GradBoost.
    Runs inside a worker process, so Bagging uses n_jobs=1 (no oversubscription).
    """
    name, seed, csv, n_est, test_frac = job
    X, y, classes = load_iris(csv)
    tr, te = stratified_split(y, test_frac, np.random.default_rng(seed))
    Xtr, Xte = standardize(X[tr], X[te])
    k = len(classes)
    if name in BASES:
        make = BASES[name]
        models = {"Single": make(), "Bagging": Bagging(make, n_est, n_jobs=1, seed=seed)}
    else:
        models = {"GradBoost": GradientBoosting(GRAD_BASES[name], n_est)}
    out = {}
    for method, model in models.items():
        res = evaluate(model, Xtr, y[tr], Xte, y[te], k)
        res.pop("pred_labels")
        out[method] = res
    return name, out


def entries(args, pool=None, n_jobs=None):
    """(base, method, model) for every row: Single+Bagging per classifier, GradBoost per regressor."""
    for name, make in BASES.items():
        yield name, "Single", make()
        yield name, "Bagging", Bagging(make, args.n_estimators, n_jobs, 42, pool)
    for name, make in GRAD_BASES.items():
        yield name, "GradBoost", GradientBoosting(make, args.n_estimators, 0.3)


def report_single_split(X, y, k, args, pool, pct):
    """Table 1: performance + speed for each method on one split (parallel ensembles)."""
    tr, te = stratified_split(y, args.test_frac, np.random.default_rng(42))
    Xtr, Xte = standardize(X[tr], X[te])
    print(f"\n== Single split {pct} (train={len(tr)}, test={len(te)}), Bagging parallel, boosting sequential ==")
    print(HEADER)
    for name, method, model in entries(args, pool, args.jobs):
        print(format_row(name, method, evaluate(model, Xtr, y[tr], Xte, y[te], k)))
    return Xtr, Xte, tr, te


def report_speedup(y, k, args, pool, split):
    """Table 2: serial (n_jobs=1) vs parallel timings; asserts identical predictions."""
    Xtr, Xte, tr, te = split
    workers = args.jobs or os.cpu_count()
    print(f"\n== Parallel vs serial speed (n_jobs=1 vs {workers}) ==")
    print(f"{'base':13s} {'method':9s} {'stage':5s} {'serial ms':>10s} {'parallel ms':>12s} {'speedup':>8s}")
    serial = {(n, m): mod for n, m, mod in entries(args, None, 1)}
    parallel = {(n, m): mod for n, m, mod in entries(args, pool, args.jobs)}
    for (name, method), ser_model in serial.items():
        if method != "Bagging":                  # Single and GradBoost have no parallel version
            continue
        ser = evaluate(ser_model, Xtr, y[tr], Xte, y[te], k)
        par = evaluate(parallel[(name, method)], Xtr, y[tr], Xte, y[te], k)
        assert np.array_equal(ser["pred_labels"], par["pred_labels"])
        for stage in ("fit", "pred"):
            s_, p_ = ser[stage] * 1e3, par[stage] * 1e3
            print(f"{name:13s} {method:9s} {stage:5s} {s_:10.2f} {p_:12.2f} {s_ / p_:7.2f}x")
    print("(Gradient boosting is sequential by design, so it is not part of this comparison.)")


def report_repeated(args, pool, pct):
    """Table 3: mean metrics over many random splits, experiments run in parallel."""
    jobs = [(n, s, args.csv, args.n_estimators, args.test_frac)
            for n in list(BASES) + list(GRAD_BASES) for s in range(args.seeds)]
    res = list(pool.map(run_one, jobs))
    print(f"\n== Mean over {args.seeds} random {pct} stratified splits (each run single-process) ==")
    print(HEADER + f" {'acc std':>8s}")
    for name in list(BASES) + list(GRAD_BASES):
        for method in ("Single", "Bagging", "GradBoost"):
            runs = [o[method] for n, o in res if n == name and method in o]
            if not runs:
                continue
            avg = {key: np.mean([m[key] for m in runs]) for key in ("acc", "prec", "rec", "f1", "fit", "pred")}
            print(format_row(name, method, avg) + f" {np.std([m['acc'] for m in runs]):8.3f}")
