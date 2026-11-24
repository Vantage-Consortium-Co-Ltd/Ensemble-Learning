"""
test_ensemble.py – ทดสอบ Bagging / AdaBoost / majority_vote ด้วยข้อมูลสังเคราะห์

การเปลี่ยนแปลงจากต้นฉบับ:
- เพิ่ม test_vote_n_class_อนุมานจาก_preds เพื่อทดสอบกรณีที่ n_class ไม่ถูกส่งเข้า
- เพิ่ม test_adaboost_predict_proba เพื่อครอบคลุม method ใหม่
- ปรับ test_bagging_สร้างต้นไม้ครบ ให้ตรวจสอบ n_class ด้วย
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from bagging import Bagging
from boosting import AdaBoost
from common import majority_vote

SEED = 0


def make_data(n: int = 300, n_class: int = 3):
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=n_class,
        n_clusters_per_class=1,
        random_state=SEED,
    )
    return X[:200], X[200:], y[:200], y[200:]


# ===== majority_vote =====

def test_vote_เสียงข้างมากชนะ():
    preds = np.array([[0, 1, 2],
                      [0, 2, 2],
                      [1, 1, 0]])
    assert majority_vote(preds).tolist() == [0, 1, 2]


def test_vote_ถ่วงน้ำหนัก_เสียงหนักชนะเสียงข้างมาก():
    preds = np.array([[0], [0], [1]])
    # 2 เสียงโหวต 0 (น้ำหนักรวม 2) แพ้ 1 เสียงโหวต 1 (น้ำหนัก 5)
    assert majority_vote(preds, np.array([1.0, 1.0, 5.0])).tolist() == [1]


def test_vote_n_class_อนุมานจาก_preds():
    """ถ้าทุกโมเดลเห็นแค่ 2 คลาส แต่ n_class=3 → ผลต้องไม่ error"""
    preds = np.array([[0, 1], [0, 1]])
    result = majority_vote(preds, n_class=3)
    assert result.tolist() == [0, 1]


# ===== Bagging =====

def test_bagging_ผลเท่ากันทั้งแบบเรียงและขนาน():
    Xtr, Xte, ytr, _ = make_data()
    tree = DecisionTreeClassifier(random_state=0)
    seq = Bagging(tree, 20, workers=1).fit(Xtr, ytr).predict(Xte)
    par = Bagging(tree, 20, workers=4).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(seq, par)


def test_bagging_สร้างต้นไม้ครบ():
    Xtr, _, ytr, _ = make_data()
    model = Bagging(DecisionTreeClassifier(), 15, workers=2).fit(Xtr, ytr)
    assert len(model.models) == 15
    assert model.n_class == 3
    # fit ไม่เก็บ X, y ไว้แล้ว (closure-based)
    assert not hasattr(model, "X")
    assert not hasattr(model, "y")


def test_bagging_แม่นกว่าเดาสุ่ม():
    Xtr, Xte, ytr, yte = make_data()
    pred = Bagging(DecisionTreeClassifier(random_state=0), 30).fit(Xtr, ytr).predict(Xte)
    assert np.mean(pred == yte) > 0.7  # 3 คลาส เดาสุ่มได้ ~0.33


# ===== AdaBoost =====

def test_adaboost_ดีกว่า_stump_ต้นเดียว():
    Xtr, Xte, ytr, yte = make_data()
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    single = np.mean(stump.fit(Xtr, ytr).predict(Xte) == yte)
    boosted = np.mean(AdaBoost(stump, 50).fit(Xtr, ytr).predict(Xte) == yte)
    assert boosted > single


def test_adaboost_alpha_เป็นบวกทุกต้น():
    Xtr, _, ytr, _ = make_data()
    model = AdaBoost(DecisionTreeClassifier(max_depth=1), 30).fit(Xtr, ytr)
    assert len(model.models) == len(model.alphas) > 0
    assert all(a > 0 for a in model.alphas)


def test_adaboost_หยุดเองเมื่อทายถูกหมด():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = AdaBoost(DecisionTreeClassifier(max_depth=1), 50).fit(X, y)
    assert len(model.models) == 1
    assert model.predict(X).tolist() == y.tolist()


def test_adaboost_predict_proba_shape():
    """predict_proba ต้องคืน shape (n_samples, n_class)"""
    Xtr, Xte, ytr, _ = make_data()
    model = AdaBoost(DecisionTreeClassifier(max_depth=1), 20).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte), model.n_class)
