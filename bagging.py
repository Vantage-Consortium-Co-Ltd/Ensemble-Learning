"""
bagging.py – Bagging: เทรนต้นไม้หลายต้นแบบขนาน (parallel) ด้วย bootstrap sampling

การเปลี่ยนแปลงจากต้นฉบับ:
- ลบ --scale ออก (ไม่จำเป็น – ทำให้สับสน)
- ใช้ closure ใน fit() แทนการเก็บ self.X / self.y → ไม่ต้อง del ท้าย fit
- เพิ่ม type hints
- ลบ global os.chdir() ออกจาก __main__ (side-effect ที่ไม่จำเป็น)

รัน: python bagging.py [--n 50] [--workers 4]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier

from common import evaluate, load_iris, majority_vote


class Bagging:
    """Bagging classifier ด้วย bootstrap + majority vote

    Parameters
    ----------
    base_model : sklearn estimator ที่จะ clone แล้วเทรนหลายตัว
    n_models   : จำนวนต้นไม้ (default 50)
    workers    : จำนวน thread – None = ใช้จำนวน CPU ทั้งหมด, 1 = sequential
    seed       : random seed ตั้งต้น (แต่ละต้นได้ seed = seed + i)
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        n_models: int = 50,
        workers: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.base_model = base_model
        self.n_models = n_models
        self.workers = workers or os.cpu_count()
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Bagging":
        self.n_class = int(y.max()) + 1

        # Closure รับ X, y โดยตรง → ไม่ต้องเก็บเป็น instance attribute
        # (ทั้งยังปลอดภัยกับ thread เพราะแต่ละ thread อ่านเท่านั้น)
        def _train_one(i: int) -> BaseEstimator:
            rng = np.random.default_rng(self.seed + i)
            idx = rng.integers(0, len(X), len(X))
            return clone(self.base_model).fit(X[idx], y[idx])

        # ต้นไม้แต่ละต้นเป็นอิสระต่อกัน → เทรนพร้อมกันได้
        # Thread เหมาะกว่า Process สำหรับ sklearn เพราะ:
        #   1. sklearn ปล่อย GIL ระหว่างเทรน (Cython/BLAS)
        #   2. ไม่ต้องเสียเวลา fork/spawn process ใหม่
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.models = list(pool.map(_train_one, range(self.n_models)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return majority_vote(preds, n_class=self.n_class)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bagging vs single Decision Tree")
    ap.add_argument("--n", type=int, default=50, help="จำนวนต้นไม้ (default 50)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="จำนวน thread – ไม่ระบุ = ใช้ทุก CPU",
    )
    args = ap.parse_args()

    Xtr, Xte, ytr, yte, _ = load_iris()
    tree = DecisionTreeClassifier(random_state=0)

    rows = [
        evaluate("Decision Tree (single)", clone(tree), Xtr, ytr, Xte, yte),
        evaluate(
            "Bagging (sequential)", Bagging(tree, args.n, workers=1), Xtr, ytr, Xte, yte
        ),
        evaluate(
            "Bagging (parallel)",
            Bagging(tree, args.n, workers=args.workers),
            Xtr,
            ytr,
            Xte,
            yte,
        ),
    ]

    res = pd.DataFrame(rows)
    print(f"\ntrain={len(Xtr)}  test={len(Xte)}  n_models={args.n}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    os.makedirs("output", exist_ok=True)
    res.to_csv("output/bagging_results.csv", index=False)
    print("\n→ บันทึก output/bagging_results.csv")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"เวลารวม {time.perf_counter() - t0:.2f}s")
