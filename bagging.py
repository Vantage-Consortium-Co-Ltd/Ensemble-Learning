"""Bagging: เทรนต้นไม้หลายต้นพร้อมกัน (parallel) แต่ละต้นเห็นข้อมูลที่สุ่มแบบใส่คืน (bootstrap) แล้วโหวต

รัน: python bagging.py [--n 50] [--workers 4] [--scale 1]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from common import load_iris, majority_vote, evaluate


class Bagging:
    def __init__(self, base_model, n_models=50, workers=None, seed=0):
        self.base_model = base_model
        self.n_models = n_models
        self.workers = workers or os.cpu_count()    # 1 = เทรนทีละต้น (sequential)
        self.seed = seed

    def _train_one(self, i):
        # สุ่มแถวแบบใส่คืน จำนวนเท่าข้อมูลเดิม -> บางแถวซ้ำ บางแถวไม่ถูกเลือก (~37%)
        rng = np.random.default_rng(self.seed + i)
        idx = rng.integers(0, len(self.X), len(self.X))
        return clone(self.base_model).fit(self.X[idx], self.y[idx])

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_class = int(y.max()) + 1
        # ต้นไม้แต่ละต้นไม่ขึ้นต่อกัน จึงแยกเทรนพร้อมกันได้
        # ใช้ thread เพราะ sklearn ปล่อย GIL ตอนเทรน และไม่ต้องเสียเวลาเปิด process ใหม่บน Windows
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.models = list(pool.map(self._train_one, range(self.n_models)))
        del self.X, self.y
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return majority_vote(preds, n_class=self.n_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50, help='จำนวนต้นไม้')
    ap.add_argument('--workers', type=int, default=None, help='จำนวน thread (ค่าเริ่มต้น = จำนวน CPU)')
    ap.add_argument('--scale', type=int, default=1, help='คูณขนาด train set กี่เท่า (Iris เล็กมาก parallel จะยังไม่เร็วกว่า)')
    args = ap.parse_args()

    Xtr, Xte, ytr, yte, _ = load_iris()
    Xtr, ytr = np.tile(Xtr, (args.scale, 1)), np.tile(ytr, args.scale)
    # Bagging ลด variance จึงใช้ต้นไม้ลึกเต็มที่ (overfit ง่ายถ้าใช้ต้นเดียว)
    tree = DecisionTreeClassifier(random_state=0)

    rows = [evaluate('Decision Tree (single)', clone(tree), Xtr, ytr, Xte, yte),
            evaluate('Bagging (sequential)', Bagging(tree, args.n, workers=1), Xtr, ytr, Xte, yte),
            evaluate('Bagging (parallel)', Bagging(tree, args.n, workers=args.workers), Xtr, ytr, Xte, yte)]

    res = pd.DataFrame(rows)
    print(f'train={len(Xtr)} test={len(Xte)} n_models={args.n}')
    print(res.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    os.makedirs('output', exist_ok=True)
    res.to_csv('output/bagging_results.csv', index=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.perf_counter()
    main()
    print(f'total {time.perf_counter() - t0:.2f}s')
