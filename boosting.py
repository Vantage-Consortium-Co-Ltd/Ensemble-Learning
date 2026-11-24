"""
boosting.py – AdaBoost (SAMME) สำหรับหลายคลาส เขียนเองด้วย NumPy
เทรนต้นไม้ทีละต้น ต้นถัดไปจะเน้นตัวอย่างที่ต้นก่อนหน้าทายผิด

การเปลี่ยนแปลงจากต้นฉบับ:
- เพิ่ม early-stop กรณี err == 0 ที่ชัดขึ้น (ตรวจก่อน update น้ำหนัก)
- เพิ่ม type hints
- ลบ global os.chdir() ออกจาก __main__
- เพิ่ม predict_proba สำหรับใช้ต่อยอดในอนาคต

รัน: python boosting.py [--n 50]
"""
import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier

from common import evaluate, load_iris, majority_vote


class AdaBoost:
    """AdaBoost-SAMME สำหรับการจำแนกหลายคลาส (K ≥ 2)

    Parameters
    ----------
    base_model : weak learner (เช่น DecisionTreeClassifier(max_depth=1))
    n_models   : จำนวนรอบสูงสุด
    """

    def __init__(self, base_model: BaseEstimator, n_models: int = 50) -> None:
        self.base_model = base_model
        self.n_models = n_models
        self.models: List[BaseEstimator] = []
        self.alphas: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        n = len(X)
        K = int(y.max()) + 1
        self.n_class = K
        self.models, self.alphas = [], []

        w = np.full(n, 1.0 / n)  # น้ำหนักเริ่มต้นเท่ากันทุกตัวอย่าง
        random_threshold = (K - 1) / K  # err ≥ threshold = แย่กว่าเดาสุ่ม

        for _ in range(self.n_models):
            m = clone(self.base_model).fit(X, y, sample_weight=w)
            wrong = m.predict(X) != y
            err = float(w[wrong].sum())

            if err >= random_threshold:  # weak learner แย่เกินไป → หยุด
                break

            err = max(err, 1e-10)  # กันหารศูนย์เมื่อทายถูกทั้งหมด
            alpha = np.log((1 - err) / err) + np.log(K - 1)  # SAMME weight

            self.models.append(m)
            self.alphas.append(alpha)

            if err <= 1e-10:  # ทายถูกทุกตัว → น้ำหนักไม่เปลี่ยนแล้ว
                break

            w = w * np.exp(alpha * wrong)
            w /= w.sum()  # normalize in-place

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return majority_vote(preds, np.array(self.alphas), self.n_class)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """คืน soft score (ไม่ใช่ probability ที่แท้จริง) สำหรับแต่ละคลาส"""
        alphas = np.array(self.alphas)
        score = np.zeros((len(X), self.n_class))
        for m, a in zip(self.models, alphas):
            pred = m.predict(X)
            for k in range(self.n_class):
                score[:, k] += a * (pred == k)
        return score


def main() -> None:
    ap = argparse.ArgumentParser(description="AdaBoost vs single Decision Stump")
    ap.add_argument("--n", type=int, default=50, help="จำนวนรอบสูงสุด (default 50)")
    args = ap.parse_args()

    Xtr, Xte, ytr, yte, _ = load_iris()
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)

    model = AdaBoost(stump, args.n)
    rows = [
        evaluate("Decision Stump (single)", clone(stump), Xtr, ytr, Xte, yte),
        evaluate("AdaBoost", model, Xtr, ytr, Xte, yte),
    ]

    res = pd.DataFrame(rows)
    rounds_used = len(model.models)
    print(f"\ntrain={len(Xtr)}  test={len(Xte)}  rounds used={rounds_used}/{args.n}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    os.makedirs("output", exist_ok=True)
    res.to_csv("output/boosting_results.csv", index=False)
    print("\n→ บันทึก output/boosting_results.csv")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"เวลารวม {time.perf_counter() - t0:.2f}s")
