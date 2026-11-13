"""Parallel helpers built on concurrent.futures.ProcessPoolExecutor."""
import os
from concurrent.futures import ProcessPoolExecutor


def parallel_map(fn, items, n_jobs=None, executor=None):
    """Map fn over items.

    * executor given -> use that (shared, already-warm) pool
    * n_jobs == 1    -> plain in-process loop (no pool overhead)
    * otherwise      -> temporary pool with n_jobs workers (None = all cores)
    """
    if executor is not None:
        return list(executor.map(fn, items))
    if n_jobs == 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(fn, items))


def _noop(_):
    return 0


def warm_pool(n_jobs=None):
    """Create a pool and force every worker to start, so later timings exclude startup."""
    pool = ProcessPoolExecutor(n_jobs)
    list(pool.map(_noop, range((n_jobs or os.cpu_count()) * 4)))
    return pool
