#!/usr/bin/env python3
"""
Train one line-level ranker per invoice field from FATURA line tables.

Inputs:
results/line/fatura_lines_train.csv
results/line/fatura_lines_dev.csv
results/line/fatura_lines_test.csv

Outputs:
models/line/<field>_line_ranker.pkl
models/line/line_vectorizer.pkl
results/line/line_ranker_training_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJ_ROOT / "results" / "line"
MODEL_DIR = PROJ_ROOT / "models" / "line"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RESULTS_DIR / "fatura_lines_train.csv"
DEV_CSV = RESULTS_DIR / "fatura_lines_dev.csv"
TEST_CSV = RESULTS_DIR / "fatura_lines_test.csv"

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    s = df["line_text"].fillna("")
    out = pd.DataFrame(index=df.index)
    out["has_digit"] = s.str.contains(r"\d", regex=True).astype(int)
    out["has_currency"] = s.str.contains(r"[$€£¥]|USD|EUR|GBP", regex=True, case=False).astype(int)
    out["has_date_word"] = s.str.contains(r"\b(date|due|invoice)\b", regex=True, case=False).astype(int)
    out["has_total_word"] = s.str.contains(r"\b(total|balance|amount|subtotal|vat|tax)\b", regex=True, case=False).astype(int)
    out["has_bill_word"] = s.str.contains(r"\b(bill|buyer|customer|client|ship)\b", regex=True, case=False).astype(int)
    out["uppercase_ratio"] = s.apply(
        lambda x: (sum(ch.isupper() for ch in x) / max(sum(ch.isalpha() for ch in x), 1))
    )
    out["digit_ratio"] = s.apply(
        lambda x: (sum(ch.isdigit() for ch in x) / max(len(x), 1))
    )
    return out


def build_numeric_features(df: pd.DataFrame) -> csr_matrix:
    base = pd.DataFrame(index=df.index)
    for col in [
        "left_rel",
        "top_rel",
        "right_rel",
        "bottom_rel",
        "width",
        "height",
        "token_count",
    ]:
        base[col] = df[col].fillna(0).astype(float)

    flags = add_text_flags(df)
    feats = pd.concat([base, flags], axis=1)
    return csr_matrix(feats.values.astype(float))


def make_design_matrix(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer | None = None,
    fit: bool = False,
):
    texts = df["line_text"].fillna("").astype(str)

    if fit:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=40000,
            lowercase=True,
        )
        x_text = vectorizer.fit_transform(texts)
    else:
        if vectorizer is None:
            raise ValueError("vectorizer is required when fit=False")
        x_text = vectorizer.transform(texts)

    x_num = build_numeric_features(df)
    x = hstack([x_text, x_num]).tocsr()
    return x, vectorizer


def doc_top1_accuracy(model, x, df: pd.DataFrame, y_true: np.ndarray) -> float:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        scores = model.predict_proba(x)[:, 1]

    work = df[["doc_id"]].copy()
    work["score"] = scores
    work["gold"] = y_true

    correct = 0
    total = 0
    for _, grp in work.groupby("doc_id", sort=False):
        best = grp.iloc[grp["score"].argmax()]
        gold_present = grp["gold"].max() > 0
        if gold_present:
            total += 1
            if int(best["gold"]) == 1:
                correct += 1

    return round(correct / total, 4) if total else 0.0


def evaluate_split(model, x, df: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(x)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "doc_top1_accuracy": doc_top1_accuracy(model, x, df, y_true),
        "positives": int(y_true.sum()),
        "rows": int(len(y_true)),
        "docs": int(df["doc_id"].nunique()),
    }
    return metrics


def main():
    for p in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)
    test_df = pd.read_csv(TEST_CSV)

    x_train, vectorizer = make_design_matrix(train_df, fit=True)
    x_dev, _ = make_design_matrix(dev_df, vectorizer=vectorizer, fit=False)
    x_test, _ = make_design_matrix(test_df, vectorizer=vectorizer, fit=False)

    joblib.dump(vectorizer, MODEL_DIR / "line_vectorizer.pkl")

    summary: Dict[str, Dict] = {}

    for field in FIELDS:
        y_col = f"y_{field}"
        if y_col not in train_df.columns:
            print(f"[WARN] Missing label column {y_col}, skipping")
            continue

        y_train = train_df[y_col].fillna(0).astype(int).values
        y_dev = dev_df[y_col].fillna(0).astype(int).values
        y_test = test_df[y_col].fillna(0).astype(int).values

        if y_train.sum() == 0:
            print(f"[WARN] No positives for {field} in training split, skipping")
            continue

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            C=2.0,
        )
        clf.fit(x_train, y_train)

        out_model = MODEL_DIR / f"{field}_line_ranker.pkl"
        joblib.dump(clf, out_model)

        summary[field] = {
            "train": evaluate_split(clf, x_train, train_df, y_train),
            "dev": evaluate_split(clf, x_dev, dev_df, y_dev),
            "test": evaluate_split(clf, x_test, test_df, y_test),
            "model_path": str(out_model),
        }

        print(
            f"[OK] {field:<15} "
            f"dev_top1={summary[field]['dev']['doc_top1_accuracy']:.4f} "
            f"test_top1={summary[field]['test']['doc_top1_accuracy']:.4f}"
        )

    summary_path = RESULTS_DIR / "line_ranker_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Summary -> {summary_path}")


if __name__ == "__main__":
    main()