#!/usr/bin/env python3
"""
fit_eval.py
============

Joins the Rust `screen` CLI's output (`screening_report.csv`) with ground
truth group labels (e.g. ds004504's `participants.tsv`, columns
`participant_id, Gender, Age, Group, MMSE` with `Group` in {A, F, C} for
Alzheimer's / Frontotemporal dementia / Control) and evaluates, with
leave-one-subject-out cross-validation, how well the unsupervised
screening features separate each pair of groups.

This is deliberately kept separate from the Rust pipeline: `screen` never
sees a label, so there is no way for group information to leak into the
k-means state definitions, the emission MLP, or the transition matrix.
Only this last, purely evaluative step touches ground truth.

IMPORTANT: this script reports how separable the *unsupervised* screening
signal is on *this* cohort. It is a research diagnostic-plausibility check,
not a validation of a clinical test -- see README.md.

Usage
-----
    python fit_eval.py \
        --screening-report ./screening_report.csv \
        --participants-tsv ./ds004504/participants.tsv \
        --subject-id-map "sub-{n:03d}=A,F,C"   # see --help for the simpler default

By default this script expects the screening report's `subject_id` column
to already be a BIDS subject id (e.g. `sub-001`) matching
`participants.tsv`'s `participant_id` column directly -- which is what
`eeg_feature_extraction.py` writes out of the box.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

GROUP_LABELS = {"A": "AD", "F": "FTD", "C": "CN"}


def load_screening_report(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    subject_ids = [r["subject_id"] for r in rows]
    feature_cols = [
        c
        for c in reader.fieldnames
        if c not in ("subject_id", "num_epochs", "num_network_states")
    ]
    X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
    return subject_ids, X, feature_cols


def load_labels(path: Path, subject_col: str = "participant_id", group_col: str = "Group") -> dict[str, str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {r[subject_col]: GROUP_LABELS.get(r[group_col], r[group_col]) for r in reader}


def standardize(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


def logreg_loso_auc(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Leave-one-subject-out AUROC for a binary logistic regression,
    using sklearn if available (preferred: proper L2-regularized solver),
    else a small hand-rolled gradient-descent fallback so this script has
    no hard dependency beyond numpy."""
    n = len(y)
    scores = np.zeros(n)
    try:
        from sklearn.linear_model import LogisticRegression

        for i in range(n):
            train = np.delete(np.arange(n), i)
            Xz, mu, sd = standardize(X, train)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xz[train], y[train])
            scores[i] = clf.predict_proba(Xz[i : i + 1])[0, 1]
    except ImportError:
        for i in range(n):
            train = np.delete(np.arange(n), i)
            Xz, mu, sd = standardize(X, train)
            w, b = _fit_logreg_gd(Xz[train], y[train])
            scores[i] = _sigmoid(Xz[i] @ w + b)

    auc = _auroc(y, scores)
    acc = float(np.mean((scores >= 0.5).astype(int) == y))
    return auc, acc


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logreg_gd(X, y, lr=0.1, l2=1e-2, iters=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U statistic form of AUROC.
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1
    rank_pos = ranks[len(neg) :]
    u = rank_pos.sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screening-report", type=Path, required=True)
    ap.add_argument("--participants-tsv", type=Path, required=True)
    ap.add_argument("--subject-col", default="participant_id")
    ap.add_argument("--group-col", default="Group")
    args = ap.parse_args()

    subject_ids, X, feature_cols = load_screening_report(args.screening_report)
    labels = load_labels(args.participants_tsv, args.subject_col, args.group_col)

    y_group = []
    keep = []
    for i, sid in enumerate(subject_ids):
        key = sid.split("_")[0]  # tolerate "sub-001_AD"-style synthetic ids
        if key in labels:
            y_group.append(labels[key])
            keep.append(i)
        else:
            print(f"warning: no label found for {sid} (looked up as {key}), excluding", file=sys.stderr)

    if len(keep) < 6:
        print("too few labeled subjects to evaluate meaningfully; check --subject-col/id matching", file=sys.stderr)
        sys.exit(1)

    X = X[keep]
    y_group = np.array(y_group)
    subject_ids = [subject_ids[i] for i in keep]

    print(f"{len(subject_ids)} labeled subjects: " + ", ".join(f"{g}={int((y_group==g).sum())}" for g in sorted(set(y_group))))
    print(f"features used: {feature_cols}\n")

    pairs = [("AD", "CN"), ("FTD", "CN"), ("AD", "FTD")]
    for a, b in pairs:
        mask = (y_group == a) | (y_group == b)
        if mask.sum() < 6 or len(set(y_group[mask])) < 2:
            print(f"{a} vs {b}: skipped (not enough subjects in this cohort)")
            continue
        y_bin = (y_group[mask] == a).astype(int)
        auc, acc = logreg_loso_auc(X[mask], y_bin)
        print(f"{a} vs {b}  (n={int(mask.sum())}):  LOSO AUROC = {auc:.3f}   LOSO accuracy = {acc:.3f}")

    print(
        "\nReminder: this is an unsupervised, exploratory network-dynamics signal evaluated "
        "on one cohort with no external validation set, held-out replication cohort, or "
        "clinical calibration. Treat AUROC/accuracy here as evidence for whether the signal "
        "is worth investigating further -- not as evidence it is fit for screening real "
        "patients. See README.md."
    )


if __name__ == "__main__":
    main()
