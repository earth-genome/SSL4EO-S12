#!/usr/bin/env python
"""Linear probe + kNN on frozen DINOv3 embeddings, with a pass/fail gate.

Consumes embeddings produced by ``extract_embeddings.py`` plus matching label
arrays, fits a frozen-feature classifier, and compares the metric to a target
baseline (e.g. the original DINO ViT-S/16 SSL4EO numbers). Exits non-zero if the
metric is below ``--baseline - --tolerance`` so it can gate the full GCP run.

Tasks:
  - ``multiclass`` (EuroSAT, So2Sat): logistic regression -> top-1 accuracy.
  - ``multilabel`` (BigEarthNet): one-vs-rest logistic regression -> micro mAP.

Inputs are ``.npy`` files:
  --train-x train.npy --train-y train_labels.npy --val-x val.npy --val-y val_labels.npy
Labels: int class ids (multiclass) or 0/1 matrix ``(N, n_classes)`` (multilabel).

README baselines (DINO ViT-S/16): BigEarthNet 90.5 mAP, EuroSAT 99.0 acc, So2Sat 62.2 acc.
"""

from __future__ import annotations

import argparse

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


def _knn_accuracy(train_x, train_y, val_x, val_y, k: int = 20) -> float:
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(train_x, train_y)
    return float((clf.predict(val_x) == val_y).mean() * 100.0)


def _linear_multiclass(train_x, train_y, val_x, val_y) -> float:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, multi_class="auto")
    clf.fit(train_x, train_y)
    return float((clf.predict(val_x) == val_y).mean() * 100.0)


def _linear_multilabel(train_x, train_y, val_x, val_y) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    n_classes = train_y.shape[1]
    scores = np.zeros((val_x.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        if train_y[:, c].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(train_x, train_y[:, c])
        scores[:, c] = clf.predict_proba(val_x)[:, 1]
    return float(average_precision_score(val_y, scores, average="micro") * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-x", required=True)
    ap.add_argument("--train-y", required=True)
    ap.add_argument("--val-x", required=True)
    ap.add_argument("--val-y", required=True)
    ap.add_argument("--task", choices=["multiclass", "multilabel"], required=True)
    ap.add_argument("--knn", action="store_true", help="Also report kNN (multiclass only)")
    ap.add_argument("--baseline", type=float, default=None, help="Target metric (%%) to gate against")
    ap.add_argument("--tolerance", type=float, default=0.5, help="Allowed shortfall below baseline (%%)")
    ap.add_argument("--name", default="probe")
    args = ap.parse_args()

    train_x = _normalize(np.load(args.train_x))
    val_x = _normalize(np.load(args.val_x))
    train_y = np.load(args.train_y)
    val_y = np.load(args.val_y)

    if args.task == "multiclass":
        metric = _linear_multiclass(train_x, train_y, val_x, val_y)
        metric_name = "top1-acc"
    else:
        metric = _linear_multilabel(train_x, train_y, val_x, val_y)
        metric_name = "micro-mAP"

    print(f"[{args.name}] linear {metric_name}: {metric:.2f}%")
    if args.knn and args.task == "multiclass":
        print(f"[{args.name}] kNN top1-acc: {_knn_accuracy(train_x, train_y, val_x, val_y):.2f}%")

    if args.baseline is not None:
        gate = args.baseline - args.tolerance
        status = "PASS" if metric >= gate else "FAIL"
        print(f"[{args.name}] baseline={args.baseline:.2f}%  gate>={gate:.2f}%  ->  {status}")
        if metric < gate:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
