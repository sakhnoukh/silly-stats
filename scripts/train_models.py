"""
Phase 2 — Train Baseline Models

Trains 3 classifiers on TF-IDF features from Phase 1:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM

Evaluates each on the validation set, saves:
  - Trained model artifacts   -> models/
  - Validation metrics table  -> results/model_comparison.csv
  - Detailed validation report-> results/validation_metrics.txt
  - Best model copy           -> models/best_model.pkl

Usage:
  python3 scripts/train_models.py
"""

import os
import time
import json
import numpy as np
import joblib
from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR    = os.path.join(PROJ_ROOT, "data", "features")
MODELS_DIR  = os.path.join(PROJ_ROOT, "models")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    X_train = load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
    X_val   = load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy"), allow_pickle=True)
    y_val   = np.load(os.path.join(FEAT_DIR, "y_val.npy"), allow_pickle=True)

    label_encoder = joblib.load(os.path.join(FEAT_DIR, "label_encoder.pkl"))
    class_names = list(label_encoder.classes_)

    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")
    print(f"Classes: {class_names}\n")

    return X_train, X_val, y_train, y_val, label_encoder, class_names


# ---------------------------------------------------------------------------
# Define models
# ---------------------------------------------------------------------------

def get_models():
    return [
        ("MultinomialNB",     MultinomialNB(alpha=1.0)),
        ("LogisticRegression", LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        )),
        ("LinearSVM",         LinearSVC(
            max_iter=2000, C=1.0, random_state=42
        )),
    ]


# ---------------------------------------------------------------------------
# Train + validate
# ---------------------------------------------------------------------------

def train_and_evaluate(models, X_train, X_val, y_train, y_val, class_names):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    reports = []

    for name, model in models:
        print(f"--- {name} ---")

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.2f}s")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}.pkl")
        joblib.dump(model, model_path)

        # Predict on validation
        y_pred = model.predict(X_val)

        # Metrics
        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_val, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")

        results.append({
            "model":           name,
            "accuracy":        round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall":    round(rec, 4),
            "macro_f1":        round(f1, 4),
            "train_time_s":    round(train_time, 3),
        })

        report = classification_report(
            y_val, y_pred, target_names=class_names, zero_division=0
        )
        reports.append((name, report))
        print()

    return results, reports


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(results, reports, class_names):
    # 1. model_comparison.csv
    import csv
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # 2. validation_metrics.txt
    txt_path = os.path.join(RESULTS_DIR, "validation_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Phase 2 — Validation Metrics\n")
        f.write("=" * 60 + "\n\n")

        # Comparison table
        f.write(f"{'Model':<25} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['model']:<25} {r['accuracy']:>8.4f} {r['macro_precision']:>10.4f} "
                    f"{r['macro_recall']:>8.4f} {r['macro_f1']:>8.4f}\n")
        f.write("\n")

        # Per-model classification reports
        for name, report in reports:
            f.write(f"\n--- {name} ---\n")
            f.write(report)
            f.write("\n")

    print(f"Saved: {txt_path}")


def select_best_model(results):
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBest model: {best['model']} (macro F1 = {best['macro_f1']:.4f})")

    # Copy best model to best_model.pkl
    src = os.path.join(MODELS_DIR, f"{best['model'].lower()}.pkl")
    dst = os.path.join(MODELS_DIR, "best_model.pkl")
    import shutil
    shutil.copy2(src, dst)
    print(f"Saved best model -> models/best_model.pkl")

    # Save selection metadata
    meta = {
        "selected_model": best["model"],
        "selection_criterion": "macro_f1",
        "validation_macro_f1": best["macro_f1"],
        "validation_accuracy": best["accuracy"],
    }
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2 — Train Baseline Models")
    print("=" * 60 + "\n")

    X_train, X_val, y_train, y_val, label_encoder, class_names = load_data()
    models = get_models()

    results, reports = train_and_evaluate(
        models, X_train, X_val, y_train, y_val, class_names
    )

    save_results(results, reports, class_names)
    best = select_best_model(results)

    print("\nDone. Next: run evaluate_model.py for final test evaluation.")


if __name__ == "__main__":
    main()
