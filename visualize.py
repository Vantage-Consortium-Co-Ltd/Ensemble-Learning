"""วาดกราฟเทียบ Bagging กับ Boosting -> output/compare.png
ต้องรัน bagging.py และ boosting.py ก่อน (แต่ละไฟล์จะเขียน csv ไว้ใน output/)

รัน: python visualize.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.concat([pd.read_csv('output/bagging_results.csv'),
                pd.read_csv('output/boosting_results.csv')]).set_index('Model')

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
df[['Accuracy', 'F1']].plot.bar(ax=ax[0], ylim=(0, 1.15), title='Test set quality')
df[['Train_s', 'Predict_s']].plot.bar(ax=ax[1], logy=True, title='Time (seconds, log scale)')
for a in ax:
    a.tick_params(axis='x', rotation=20)
    for bars in a.containers:
        a.bar_label(bars, fmt='%.3g', fontsize=8)
fig.tight_layout()
fig.savefig('output/compare.png', dpi=150)
print('saved output/compare.png')
