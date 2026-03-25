"""
Phase 3 — Train Classical ML Field Rankers

Trains one binary classifier per field on candidate tables.

Expected input CSVs:
  - results/candidates_recipient_name_labeled.csv
  - results/candidates_invoice_number_labeled.csv
  - results/candidates_total_amount_labeled.csv

Each CSV must contain:
  - all candidate feature columns
  - label column with values {0,1}

Outputs:
  - models/recipient_name_ranker.pkl
  - models/invoice_number_ranker.pkl
  - models/total_amount_ranker.pkl
  - results/ranker_training_summary.txt

Usage:
    python scripts/train_field_rankers.py
"""

from pathlib import Path
import pickle
import json
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "ml"
CANDIDATES_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"
MODELS_DIR = PROJ_ROOT / "models" / "phase3_extraction"

FIELD_CONFIG = {
    "recipient_name": {
        "input_csv": CANDIDATES_DIR / "candidates_recipient_name_labeled.csv",
        "model_path": MODELS_DIR / "recipient_name_ranker.pkl",
    },
    "invoice_number": {
        "input_csv": CANDIDATES_DIR / "candidates_invoice_number_labeled.csv",
        "model_path": MODELS_DIR / "invoice_number_ranker.pkl",
    },
    "total_amount": {
        "input_csv": CANDIDATES_DIR / "candidates_total_amount_labeled.csv",
        "model_path": MODELS_DIR / "total_amount_ranker.pkl",
    },
}


# ---------------------------------------------------------------------------
# Feature handling
# ---------------------------------------------------------------------------

META_COLUMNS = {
    "doc_id", "file_name", "_split", "field", "candidate_text",
    "label", "notes"
}


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Split usable feature columns into numeric vs text/categorical.
    We keep candidate_text out for v1 to reduce overfitting to exact strings.
    """
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_feature_dicts(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    rows = []
    for _, row in df.iterrows():
        feat = {}

        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                feat[col] = 0
            else:
                feat[col] = float(val)

        for col in categorical_cols:
            val = row[col]
            if pd.isna(val):
                feat[col] = "__MISSING__"
            else:
                feat[col] = str(val)

        rows.append(feat)
    return rows


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_field(field_name: str, csv_path: str, model_path: str):
    print(f"\n=== Training ranker for {field_name} ===")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if df["label"].nunique() < 2:
        raise ValueError(f"{field_name}: need both positive and negative labels")

    # Group split by document so candidates from same doc don't leak
    groups = df["doc_id"].astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    numeric_cols, categorical_cols = infer_feature_columns(df)

    X_train_dict = make_feature_dicts(train_df, numeric_cols, categorical_cols)
    X_test_dict = make_feature_dicts(test_df, numeric_cols, categorical_cols)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # v1: Logistic Regression for generalization
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )),
    ])

    pipeline.fit(X_train_dict, y_train)
    y_pred = pipeline.predict(X_test_dict)
    y_prob = pipeline.predict_proba(X_test_dict)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "field_name": field_name,
            "pipeline": pipeline,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }, f)

    print(f"Train docs: {train_df['doc_id'].nunique()} | Test docs: {test_df['doc_id'].nunique()}")
    print(f"Labeled rows: {len(df)} | Positives: {int(df['label'].sum())}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1:        {f1:.3f}")

    return {
        "field": field_name,
        "train_docs": int(train_df["doc_id"].nunique()),
        "test_docs": int(test_df["doc_id"].nunique()),
        "rows": int(len(df)),
        "positives": int(df["label"].sum()),
        "accuracy": float(round(acc, 4)),
        "precision": float(round(p, 4)),
        "recall": float(round(r, 4)),
        "f1": float(round(f1, 4)),
        "report": report,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3 — Train Field Rankers")
    print("=" * 60)

    summaries = []

    for field_name, cfg in FIELD_CONFIG.items():
        summary = train_one_field(
            field_name=field_name,
            csv_path=cfg["input_csv"],
            model_path=cfg["model_path"],
        )
        summaries.append(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_txt = RESULTS_DIR / "ranker_training_summary.txt"
    summary_json = RESULTS_DIR / "ranker_training_summary.json"

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Phase 3 — Field Ranker Training Summary\n")
        f.write("=" * 50 + "\n\n")
        for s in summaries:
            f.write(f"[{s['field']}]\n")
            f.write(f"Train docs: {s['train_docs']}\n")
            f.write(f"Test docs: {s['test_docs']}\n")
            f.write(f"Labeled rows: {s['rows']}\n")
            f.write(f"Positives: {s['positives']}\n")
            f.write(f"Accuracy: {s['accuracy']}\n")
            f.write(f"Precision: {s['precision']}\n")
            f.write(f"Recall: {s['recall']}\n")
            f.write(f"F1: {s['f1']}\n")
            f.write("\nClassification report:\n")
            f.write(s["report"])
            f.write("\n" + "-" * 50 + "\n\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n✓ {summary_txt}")
    print(f"✓ {summary_json}")
    print("\nDone. Next step: use the trained rankers in extract_invoices_ml.py")


if __name__ == "__main__":
    main()