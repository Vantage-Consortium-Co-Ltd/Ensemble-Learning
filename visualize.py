"""Plot Bagging vs Boosting accuracy/F1 and fit/predict time into results.png.

Run Bagging/bagging_iris.py and Boosting/boosting_iris.py first (each writes a results.csv), then:
    python visualize.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).parent
df = pd.concat([pd.read_csv(root / d / "results.csv").assign(Method=lambda x: d + "\n" + x.Method)
                for d in ("Bagging", "Boosting")]).set_index("Method")  # both CSVs have a "KNN (single)" row

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
df[["Acc", "F1"]].plot.bar(ax=ax[0], ylim=(0, 1.2), title="Prediction quality (test set)")
ax[0].legend(loc="upper center", ncol=2)
df[["Fit_s", "Predict_s"]].plot.bar(ax=ax[1], logy=True, title="Speed in seconds (log scale)")
for a in ax:
    a.tick_params(axis="x", rotation=15)
    for c in a.containers:
        a.bar_label(c, fmt="%.3g", fontsize=8)
fig.tight_layout()
fig.savefig(root / "results.png", dpi=150)
print("saved", root / "results.png")
