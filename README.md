# Ensemble Learning: Bagging & AdaBoost (NumPy only)

Bagging and AdaBoost (multi-class SAMME) written from scratch with **NumPy**, run in
**parallel** with `concurrent.futures`. Base classifiers (KNN, Gaussian Naive Bayes,
decision trees) come from scikit-learn. Dataset: the Kaggle
[Iris dataset](https://www.kaggle.com/datasets/vikrishnan/iris-dataset).

## Layout

```
data/iris.data.csv            dataset (4 features + species, no header)
ensemble_learning/
  data.py                     load CSV, stratified split, standardise      (NumPy)
  metrics.py                  confusion matrix, acc/precision/recall/F1, timer (NumPy)
  voting.py                   weighted majority vote                        (NumPy)
  parallel.py                 process-pool helpers
  bagging.py                  Bagging
  boosting.py                 AdaBoost (SAMME)
  classifiers.py              base classifiers (scikit-learn)
  experiments.py              experiment runners + report tables
run.py                        command-line entry point
```

## Run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --n-estimators 25 --test-frac 0.2 --seeds 20 --jobs 8
```

Prints three tables: performance and speed on one split, serial vs parallel speedup,
and mean metrics over many random splits.

## Use as a library

```python
from ensemble_learning import Bagging, AdaBoost
from ensemble_learning.classifiers import tree

model = Bagging(tree, n_estimators=25, n_jobs=4).fit(X_train, y_train)
pred = model.predict(X_test)
```

## How parallelism is used

| Stage | Parallel? |
|---|---|
| Bagging fit / predict | yes, estimators are independent |
| AdaBoost fit | no, each round depends on the previous one |
| AdaBoost predict | yes |
| Repeated experiments | yes, one task per (base, seed) |

For tiny models and data like Iris, process start-up and data transfer can outweigh the
gain, so predict can be slower in parallel. Parallelism pays off with heavier base models.
