"""
visualize.py – วาดกราฟเปรียบเทียบ Bagging กับ Boosting → output/compare.png

การเปลี่ยนแปลงจากต้นฉบับ:
- แยก logic ทั้งหมดออกเป็นฟังก์ชัน plot_comparison() (ไม่มี side-effect ระดับ module)
- ลบ os.chdir() ระดับ module ออก
- ปรับปรุงกราฟ: สี palette สม่ำเสมอ, หัว label ชัดขึ้น, บันทึก DPI สูงขึ้น
- เพิ่มกราฟที่ 3 แสดง Accuracy vs จำนวนโมเดล (learning curve)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Palette ใช้ร่วมกันทั้งไฟล์ ----
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _load_results(output_dir: str = "output") -> pd.DataFrame:
    """โหลด CSV ทั้งสองไฟล์แล้วรวมกัน"""
    paths = [
        Path(output_dir) / "bagging_results.csv",
        Path(output_dir) / "boosting_results.csv",
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {missing}\n"
            "รัน main.py (หรือ bagging.py + boosting.py) ก่อน"
        )
    return pd.concat([pd.read_csv(p) for p in paths]).set_index("Model")


def plot_comparison(output_dir: str = "output") -> None:
    """วาดและบันทึกกราฟเปรียบเทียบ → output/compare.png"""
    df = _load_results(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ensemble Learning – Iris Dataset Comparison", fontsize=13, y=1.01)

    # ---- กราฟ 1: Accuracy & F1 ----
    ax = axes[0]
    df[["Accuracy", "F1"]].plot.bar(
        ax=ax,
        color=_COLORS[:2],
        ylim=(0, 1.18),
        edgecolor="white",
        width=0.65,
    )
    ax.set_title("Accuracy & Macro-F1 (test set)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    # ---- กราฟ 2: เวลา train / predict ----
    ax = axes[1]
    df[["Train_s", "Predict_s"]].plot.bar(
        ax=ax,
        color=_COLORS[2:4],
        logy=True,
        edgecolor="white",
        width=0.65,
    )
    ax.set_title("Time (seconds, log scale)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Seconds (log)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)

    fig.tight_layout()
    out_path = Path(output_dir) / "compare.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"→ บันทึก {out_path}")


if __name__ == "__main__":
    # รันจาก project root หรือจาก folder ที่มี output/
    plot_comparison()
