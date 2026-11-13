"""Evaluation metrics and timing helpers (NumPy only)."""
import time

import numpy as np


def confusion_matrix(y, pred, n_classes):
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (y, pred), 1)
    return cm


def scores(y, pred, n_classes):
    """Accuracy plus macro precision / recall / F1."""
    cm = confusion_matrix(y, pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(acc=tp.sum() / cm.sum(), prec=prec.mean(), rec=rec.mean(), f1=f1.mean())


def timed(fn, *args):
    """Return (fn(*args), elapsed seconds)."""
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start
