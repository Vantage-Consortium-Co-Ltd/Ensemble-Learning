"""Command-line entry point:  python run.py [--test-frac 0.2] [--n-estimators 25] ..."""
import argparse
import os
import time

from ensemble_learning.data import load_iris
from ensemble_learning.experiments import report_repeated, report_single_split, report_speedup
from ensemble_learning.parallel import warm_pool

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "iris.data.csv"))
    ap.add_argument("--n-estimators", type=int, default=25)
    ap.add_argument("--test-frac", type=float, default=0.2, help="test fraction (0.2 = 80/20 split)")
    ap.add_argument("--seeds", type=int, default=20, help="number of random splits for the averaged table")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    return ap.parse_args()


def main():
    args = parse_args()
    pct = f"{round((1 - args.test_frac) * 100)}/{round(args.test_frac * 100)}"
    X, y, classes = load_iris(args.csv)
    workers = args.jobs or os.cpu_count()
    print(f"Loaded {args.csv}: {X.shape[0]} samples, {X.shape[1]} features, classes={[str(c) for c in classes]}")
    print(f"CPU cores: {os.cpu_count()}, workers: {workers}, estimators: {args.n_estimators}")

    start = time.perf_counter()
    pool = warm_pool(args.jobs)    # one shared pool so timings exclude process startup
    print(f"Process-pool startup (one-off, excluded from timings below): "
          f"{(time.perf_counter() - start) * 1e3:.0f} ms")
    try:
        split = report_single_split(X, y, len(classes), args, pool, pct)
        report_speedup(y, len(classes), args, pool, split)
        report_repeated(args, pool, pct)
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
