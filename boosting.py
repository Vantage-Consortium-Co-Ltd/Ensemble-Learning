"""Boosting (AdaBoost แบบ SAMME สำหรับหลายคลาส) เขียนเองด้วย NumPy
เทรนต้นไม้ทีละต้น ต้นถัดไปจะเน้นตัวอย่างที่ต้นก่อนหน้าทายผิด

รัน: python boosting.py [--n 50]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from common import load_iris, majority_vote, evaluate


class AdaBoost:
    def __init__(self, base_model, n_models=50):
        self.base_model = base_model
        self.n_models = n_models

    def fit(self, X, y):
        n, K = len(X), int(y.max()) + 1
        self.n_class = K
        w = np.full(n, 1 / n)                       # เริ่มต้นทุกตัวอย่างสำคัญเท่ากัน
        self.models, self.alphas = [], []

        for _ in range(self.n_models):
            m = clone(self.base_model).fit(X, y, sample_weight=w)
            wrong = m.predict(X) != y
            err = w[wrong].sum()

            if err >= 1 - 1 / K:                    # แย่กว่าเดาสุ่ม -> หยุด
                break
            err = max(err, 1e-10)                   # กันหารศูนย์ตอนทายถูกหมด
            alpha = np.log((1 - err) / err) + np.log(K - 1)   # ต้นที่ผิดน้อย ได้เสียงโหวตมาก

            self.models.append(m)
            self.alphas.append(alpha)
            if err <= 1e-10:                        # ทายถูกหมดแล้ว น้ำหนักจะไม่เปลี่ยนอีก
                break

            w = w * np.exp(alpha * wrong)           # เพิ่มน้ำหนักตัวที่ทายผิด
            w = w / w.sum()
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return majority_vote(preds, np.array(self.alphas), self.n_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50, help='จำนวนรอบ (ต้นไม้)')
    args = ap.parse_args()

    Xtr, Xte, ytr, yte, _ = load_iris()
    # Boosting ลด bias จึงใช้ weak learner: ต้นไม้ชั้นเดียว (decision stump)
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)

    model = AdaBoost(stump, args.n)
    rows = [evaluate('Decision Stump (single)', clone(stump), Xtr, ytr, Xte, yte),
            evaluate('AdaBoost', model, Xtr, ytr, Xte, yte)]

    res = pd.DataFrame(rows)
    print(f'train={len(Xtr)} test={len(Xte)} rounds used={len(model.models)}/{args.n}')
    print(res.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    os.makedirs('output', exist_ok=True)
    res.to_csv('output/boosting_results.csv', index=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
