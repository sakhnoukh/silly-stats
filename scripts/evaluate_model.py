"""
Phase 2 — Final Test Evaluation

Loads the best model selected during training, evaluates it once on the
held-out test set, and produces:
  - results/test_metrics.txt
  - results/classification_report.txt
  - results/confusion_matrix.png

Usage:
  python3 scripts/evaluate_model.py
"""

import os
import json
import numpy as np
import joblib
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR    = os.path.join(PROJ_ROOT, "data", "features")
MODELS_DIR  = os.path.join(PROJ_ROOT, "models")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")


# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------

def load_test_data():
    X_test = load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(FEAT_DIR, "y_test.npy"), allow_pickle=True)
    label_encoder = joblib.load(os.path.join(FEAT_DIR, "label_encoder.pkl"))
    class_names = list(label_encoder.classes_)
    print(f"X_test: {X_test.shape}  classes: {class_names}")
    return X_test, y_test, label_encoder, class_names


def load_best_model():
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    model = joblib.load(model_path)
    print(f"Loaded best model: {meta['selected_model']} "
          f"(val macro F1 = {meta['validation_macro_f1']:.4f})")
    return model, meta


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, class_names, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n=== Test Set Results ({meta['selected_model']}) ===")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall:    {rec:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    # --- test_metrics.txt ---
    txt_path = os.path.join(RESULTS_DIR, "test_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Phase 2 — Final Test Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {meta['selected_model']}\n")
        f.write(f"Selection criterion: {meta['selection_criterion']}\n")
        f.write(f"Validation macro F1: {meta['validation_macro_f1']:.4f}\n\n")
        f.write("Test Results\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:        {acc:.4f}\n")
        f.write(f"Macro Precision: {prec:.4f}\n")
        f.write(f"Macro Recall:    {rec:.4f}\n")
        f.write(f"Macro F1:        {f1:.4f}\n")
    print(f"\nSaved: {txt_path}")

    # --- classification_report.txt ---
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   zero_division=0)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {meta['selected_model']} (Test Set)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Saved: {report_path}")
    print(f"\n{report}")

    # --- confusion_matrix.png ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {meta['selected_model']} (Test Set)")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    return y_pred, acc, prec, rec, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2 — Final Test Evaluation")
    print("=" * 60 + "\n")

    X_test, y_test, label_encoder, class_names = load_test_data()
    model, meta = load_best_model()
    evaluate(model, X_test, y_test, class_names, meta)

    print("\nDone. Next: run error_analysis.py")


if __name__ == "__main__":
    main()
