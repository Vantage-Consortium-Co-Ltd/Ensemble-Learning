import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

URL = 'https://huggingface.co/datasets/antoniopelusi/iris/resolve/main/iris_train.csv'


def load_iris():
    """โหลด Iris จาก Hugging Face แล้วแบ่ง train/test 80/20 (stratify ให้ทุกพันธุ์มีสัดส่วนเท่ากัน)"""
    df = pd.read_csv(URL)
    X = df.drop(columns=['species']).to_numpy(float)
    y, names = pd.factorize(df['species'])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return Xtr, Xte, ytr, yte, list(names)


def majority_vote(preds, weights=None, n_class=3):
    """preds: (จำนวนโมเดล, จำนวนตัวอย่าง) -> คลาสที่ได้คะแนนโหวตมากสุดของแต่ละตัวอย่าง
    weights = None คือโหวตเสียงละ 1 (Bagging), ถ้าส่งมาคือโหวตแบบถ่วงน้ำหนัก (Boosting)"""
    if weights is None:
        weights = np.ones(len(preds))
    score = np.zeros((preds.shape[1], n_class))
    for p, w in zip(preds, weights):
        score[np.arange(len(p)), p] += w
    return score.argmax(axis=1)


def evaluate(name, model, Xtr, ytr, Xte, yte):
    """เทรน + ทำนาย แล้วคืนผล accuracy, macro F1 และเวลาที่ใช้"""
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t1 = time.perf_counter()
    pred = model.predict(Xte)
    t2 = time.perf_counter()
    return {'Model': name,
            'Accuracy': accuracy_score(yte, pred),
            'F1': f1_score(yte, pred, average='macro'),
            'Train_s': t1 - t0,
            'Predict_s': t2 - t1}
