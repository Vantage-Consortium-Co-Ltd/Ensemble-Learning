"""
main.py – จุดเริ่มต้นเดียวที่รัน Bagging + Boosting + สร้างกราฟและตารางเปรียบเทียบ
(ไฟล์ใหม่ ไม่มีในต้นฉบับ)

รัน: python main.py [--n 50] [--workers 4]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from bagging import Bagging
from boosting import AdaBoost
from common import evaluate, load_iris
from visualize import plot_comparison

DIVIDER = "=" * 60


def print_section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ensemble Learning – Bagging & Boosting บน Iris"
    )
    ap.add_argument("--n", type=int, default=50, help="จำนวนโมเดล/รอบ (default 50)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="จำนวน thread สำหรับ Bagging (default = จำนวน CPU)",
    )
    args = ap.parse_args()

    t_total = time.perf_counter()
    os.makedirs("output", exist_ok=True)

    # ---- โหลดข้อมูล ----
    print_section("โหลดข้อมูล Iris")
    Xtr, Xte, ytr, yte, class_names = load_iris()
    print(f"train={len(Xtr)}  test={len(Xte)}  classes={class_names}")

    # ---- Bagging ----
    print_section(f"Bagging  (n_models={args.n})")
    tree = DecisionTreeClassifier(random_state=0)

    bag_rows = [
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
    bag_df = pd.DataFrame(bag_rows)
    print(bag_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bag_df.to_csv("output/bagging_results.csv", index=False)

    # ---- Boosting ----
    print_section(f"AdaBoost-SAMME  (max_rounds={args.n})")
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    ada = AdaBoost(stump, args.n)

    boost_rows = [
        evaluate("Decision Stump (single)", clone(stump), Xtr, ytr, Xte, yte),
        evaluate("AdaBoost", ada, Xtr, ytr, Xte, yte),
    ]
    boost_df = pd.DataFrame(boost_rows)
    print(f"rounds used = {len(ada.models)}/{args.n}")
    print(boost_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    boost_df.to_csv("output/boosting_results.csv", index=False)

    # ---- ตารางรวม ----
    print_section("สรุปเปรียบเทียบทุกโมเดล")
    all_df = pd.concat([bag_df, boost_df], ignore_index=True)

    # หาแถวที่ดีที่สุด
    best_acc_idx = all_df["Accuracy"].idxmax()
    best_f1_idx  = all_df["F1"].idxmax()

    print(all_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n★ Accuracy สูงสุด : {all_df.loc[best_acc_idx, 'Model']}  "
          f"({all_df.loc[best_acc_idx, 'Accuracy']:.4f})")
    print(f"★ F1 สูงสุด       : {all_df.loc[best_f1_idx, 'Model']}  "
          f"({all_df.loc[best_f1_idx, 'F1']:.4f})")

    # ---- Speedup Bagging parallel vs sequential ----
    seq_t  = all_df.loc[all_df["Model"] == "Bagging (sequential)", "Train_s"].values[0]
    par_t  = all_df.loc[all_df["Model"] == "Bagging (parallel)",   "Train_s"].values[0]
    if par_t > 0:
        speedup = seq_t / par_t
        print(f"\n⚡ Bagging speedup (parallel/sequential): {speedup:.2f}×")

    # ---- วาดกราฟ ----
    print_section("สร้างกราฟ")
    plot_comparison("output")

    print(f"\n{DIVIDER}")
    print(f"  เสร็จสิ้น – เวลารวม {time.perf_counter() - t_total:.2f}s")
    print(DIVIDER)


if __name__ == "__main__":
    main()
