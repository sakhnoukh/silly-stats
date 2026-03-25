"""
Phase 2 — Error Analysis

Collects misclassified samples from the test set, summarizes failure patterns,
and writes:
  - results/error_analysis.md

Usage:
  python3 scripts/error_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
from collections import Counter

from sklearn.metrics import confusion_matrix

PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR    = os.path.join(PROJ_ROOT, "data", "features")
PROC_DIR    = os.path.join(PROJ_ROOT, "data", "processed")
MODELS_DIR  = os.path.join(PROJ_ROOT, "models")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

MAX_SNIPPET_LEN = 200  # chars to show per misclassified sample


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all():
    X_test = load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(FEAT_DIR, "y_test.npy"), allow_pickle=True)
    label_encoder = joblib.load(os.path.join(FEAT_DIR, "label_encoder.pkl"))
    class_names = list(label_encoder.classes_)

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_meta.json")) as f:
        meta = json.load(f)

    test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

    return X_test, y_test, label_encoder, class_names, model, meta, test_df


# ---------------------------------------------------------------------------
# Analyse errors
# ---------------------------------------------------------------------------

def analyse(X_test, y_test, class_names, model, meta, test_df):
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Find misclassified indices
    wrong_mask = y_pred != y_test
    wrong_idx = np.where(wrong_mask)[0]
    n_wrong = len(wrong_idx)
    n_total = len(y_test)

    print(f"Misclassified: {n_wrong}/{n_total} ({100*n_wrong/n_total:.1f}%)\n")

    # Confusion pairs
    pair_counts = Counter()
    for i in wrong_idx:
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred[i]]
        pair_counts[(true_label, pred_label)] += 1

    # Build error examples
    examples = []
    for i in wrong_idx:
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred[i]]

        # Get text snippet from the test CSV
        if i < len(test_df):
            row = test_df.iloc[i]
            text = str(row.get("clean_text", ""))[:MAX_SNIPPET_LEN]
            file_name = row.get("file_name", "unknown")
        else:
            text = "(not available)"
            file_name = "unknown"

        examples.append({
            "index": int(i),
            "file_name": file_name,
            "true_label": true_label,
            "predicted": pred_label,
            "snippet": text,
        })

    return cm, pair_counts, examples, n_wrong, n_total


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

def write_report(cm, class_names, pair_counts, examples, n_wrong, n_total, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "error_analysis.md")

    with open(report_path, "w") as f:
        f.write("# Error Analysis — Phase 2\n\n")
        f.write(f"**Model:** {meta['selected_model']}  \n")
        f.write(f"**Evaluated on:** test set ({n_total} samples)  \n")
        f.write(f"**Misclassified:** {n_wrong} ({100*n_wrong/n_total:.1f}%)\n\n")

        # Confusion matrix as markdown table
        f.write("## Confusion Matrix\n\n")
        f.write("| True \\ Predicted | " + " | ".join(class_names) + " |\n")
        f.write("|" + "---|" * (len(class_names) + 1) + "\n")
        for i, row_name in enumerate(class_names):
            row_vals = " | ".join(str(cm[i][j]) for j in range(len(class_names)))
            f.write(f"| **{row_name}** | {row_vals} |\n")
        f.write("\n")

        # Most confused pairs
        f.write("## Most Confused Class Pairs\n\n")
        if pair_counts:
            f.write("| True Label | Predicted As | Count |\n")
            f.write("|---|---|---:|\n")
            for (true, pred), cnt in pair_counts.most_common(10):
                f.write(f"| {true} | {pred} | {cnt} |\n")
        else:
            f.write("No misclassifications found.\n")
        f.write("\n")

        # Per-class accuracy
        f.write("## Per-Class Accuracy\n\n")
        f.write("| Class | Correct | Total | Accuracy |\n")
        f.write("|---|---:|---:|---:|\n")
        for i, name in enumerate(class_names):
            total = cm[i].sum()
            correct = cm[i][i]
            acc = correct / total if total > 0 else 0
            f.write(f"| {name} | {correct} | {total} | {acc:.1%} |\n")
        f.write("\n")

        # Analysis questions from PHASE_2.md
        f.write("## Key Questions\n\n")

        # Invoice vs receipt confusion
        inv_as_rec = cm[class_names.index("invoice")][class_names.index("receipt")]
        rec_as_inv = cm[class_names.index("receipt")][class_names.index("invoice")]
        f.write(f"**Are invoices and receipts being mixed up?**  \n")
        f.write(f"Invoice → Receipt: {inv_as_rec}, Receipt → Invoice: {rec_as_inv}\n\n")

        # Form vs invoice confusion
        form_as_inv = cm[class_names.index("form")][class_names.index("invoice")]
        inv_as_form = cm[class_names.index("invoice")][class_names.index("form")]
        f.write(f"**Are forms being mistaken for invoices?**  \n")
        f.write(f"Form → Invoice: {form_as_inv}, Invoice → Form: {inv_as_form}\n\n")

        # Sample misclassified examples
        f.write("## Misclassified Examples\n\n")
        if examples:
            for ex in examples[:15]:  # show up to 15
                f.write(f"### {ex['file_name']}\n")
                f.write(f"- **True:** {ex['true_label']}  \n")
                f.write(f"- **Predicted:** {ex['predicted']}  \n")
                f.write(f"- **Snippet:** `{ex['snippet']}`\n\n")
        else:
            f.write("No misclassified examples.\n")

    print(f"Saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2 — Error Analysis")
    print("=" * 60 + "\n")

    X_test, y_test, label_encoder, class_names, model, meta, test_df = load_all()
    cm, pair_counts, examples, n_wrong, n_total = analyse(
        X_test, y_test, class_names, model, meta, test_df
    )
    write_report(cm, class_names, pair_counts, examples, n_wrong, n_total, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
