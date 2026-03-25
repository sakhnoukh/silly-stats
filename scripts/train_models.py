"""
Phase 2 — Train Classification Models with Dimensionality Reduction

Trains 4 classifiers on TF-IDF features from Phase 1:
  1. Logistic Regression
  2. Random Forest
  3. Linear SVM
  4. Multinomial Naive Bayes

For each classifier, compares:
  - Baseline: directly on TF-IDF features
  - PCA: with PCA dimensionality reduction (95% variance)
  - SVD: with Truncated SVD (95% variance)

Evaluates on validation set with:
  - Cross-validation (5-fold)
  - Confusion matrices
  - Per-class metrics (precision, recall, F1)
  - Macro and weighted averages

Saves:
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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
# Define models with and without dimensionality reduction
# ---------------------------------------------------------------------------

def get_model_configs():
    """
    Returns list of (model_name, model_obj) tuples.
    Includes baseline, PCA, and SVD variants.
    """
    configs = []

    # Base models
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    svm = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    nb = MultinomialNB(alpha=1.0)

    # Baseline (no reduction)
    configs.append(("LogisticRegression (Baseline)", lr))
    configs.append(("RandomForest (Baseline)", rf))
    configs.append(("LinearSVM (Baseline)", svm))
    configs.append(("MultinomialNB (Baseline)", nb))

    # PCA variant (150 components, or 95% whichever is smaller)
    configs.append((
        "LogisticRegression (PCA)",
        Pipeline([
            ("pca", PCA(n_components=min(150, 724), random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42))
        ])
    ))
    configs.append((
        "RandomForest (PCA)",
        Pipeline([
            ("pca", PCA(n_components=min(150, 724), random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    ))
    configs.append((
        "LinearSVM (PCA)",
        Pipeline([
            ("pca", PCA(n_components=min(150, 724), random_state=42)),
            ("svm", LinearSVC(max_iter=2000, C=1.0, random_state=42))
        ])
    ))

    # SVD variant (150 components) - suitable for sparse matrices
    configs.append((
        "LogisticRegression (SVD)",
        Pipeline([
            ("svd", TruncatedSVD(n_components=min(150, 723), random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42))
        ])
    ))
    configs.append((
        "RandomForest (SVD)",
        Pipeline([
            ("svd", TruncatedSVD(n_components=min(150, 723), random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    ))
    configs.append((
        "LinearSVM (SVD)",
        Pipeline([
            ("svd", TruncatedSVD(n_components=min(150, 723), random_state=42)),
            ("svm", LinearSVC(max_iter=2000, C=1.0, random_state=42))
        ])
    ))

    return configs


# ---------------------------------------------------------------------------
# Train + validate with cross-validation and confusion matrices
# ---------------------------------------------------------------------------

def train_and_evaluate(model_configs, X_train, X_val, y_train, y_val, class_names):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    confusion_matrices = {}
    cv_scores = {}
    trained_models = {}  # Store trained models for later use

    print("=" * 80)
    print("CROSS-VALIDATION (5-fold on training set)")
    print("=" * 80)

    for name, model in model_configs:
        print(f"\n{name}")
        print("-" * 80)

        # Cross-validation on training set
        try:
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=5,
                scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                n_jobs=-1
            )
            cv_acc = cv_results["test_accuracy"].mean()
            cv_f1 = cv_results["test_f1_macro"].mean()
            cv_scores[name] = {
                "cv_accuracy_mean": round(cv_acc, 4),
                "cv_accuracy_std": round(cv_results["test_accuracy"].std(), 4),
                "cv_f1_mean": round(cv_f1, 4),
                "cv_f1_std": round(cv_results["test_f1_macro"].std(), 4),
            }
            print(f"  CV Accuracy: {cv_acc:.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
            print(f"  CV F1 (macro): {cv_f1:.4f} (+/- {cv_results['test_f1_macro'].std():.4f})")
        except Exception as e:
            print(f"  CV failed: {e}")
            cv_scores[name] = {}

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        trained_models[name] = model  # Store trained model
        print(f"  Training time: {train_time:.2f}s")

        # Save model
        try:
            model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl")
            joblib.dump(model, model_path)
        except:
            pass  # Some pipeline models may not serialize cleanly

        # Predict on validation
        y_pred = model.predict(X_val)

        # Metrics on validation set
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        # Weighted average (accounts for class imbalance)
        prec_weighted = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        print(f"\n  Validation Accuracy: {acc:.4f}")
        print(f"  Validation F1 (macro):   {f1:.4f}")
        print(f"  Validation F1 (weighted): {f1_weighted:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=range(len(class_names)))
        confusion_matrices[name] = cm

        results.append({
            "model": name,
            "val_accuracy": round(acc, 4),
            "val_precision_macro": round(prec, 4),
            "val_recall_macro": round(rec, 4),
            "val_f1_macro": round(f1, 4),
            "val_precision_weighted": round(prec_weighted, 4),
            "val_recall_weighted": round(rec_weighted, 4),
            "val_f1_weighted": round(f1_weighted, 4),
            "train_time_s": round(train_time, 3),
            **cv_scores.get(name, {})
        })

    return results, confusion_matrices, cv_scores, trained_models


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(results, confusion_matrices, class_names):
    # 1. model_comparison.csv
    import csv
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # 2. validation_metrics.txt
    txt_path = os.path.join(RESULTS_DIR, "validation_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("Phase 2 — Validation Metrics\n")
        f.write("=" * 100 + "\n\n")

        # Comparison table
        f.write("SUMMARY TABLE (sorted by Macro F1)\n")
        f.write("-" * 100 + "\n")
        sorted_results = sorted(results, key=lambda r: r["val_f1_macro"], reverse=True)
        
        f.write(f"{'Model':<40} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 (weighted)':>14} {'Train (s)':>10}\n")
        f.write("-" * 100 + "\n")
        for r in sorted_results:
            f.write(f"{r['model']:<40} {r['val_accuracy']:>10.4f} {r['val_f1_macro']:>12.4f} "
                    f"{r['val_f1_weighted']:>14.4f} {r['train_time_s']:>10.3f}\n")
        f.write("\n")

        # Confusion matrices
        f.write("=" * 100 + "\n")
        f.write("CONFUSION MATRICES (Row = True, Col = Predicted)\n")
        f.write("=" * 100 + "\n\n")
        for model_name in [r["model"] for r in sorted_results]:
            if model_name in confusion_matrices:
                cm = confusion_matrices[model_name]
                f.write(f"{model_name}\n")
                f.write("-" * 80 + "\n")
                f.write("                ")
                for cls in class_names:
                    f.write(f"{cls:>12}")
                f.write("\n")
                for i, true_cls in enumerate(class_names):
                    f.write(f"{true_cls:>16}")
                    for j in range(len(class_names)):
                        f.write(f"{cm[i, j]:>12}")
                    f.write("\n")
                f.write("\n")

        # Per-model detailed metrics
        f.write("=" * 100 + "\n")
        f.write("DETAILED METRICS (Precision, Recall, F1 per class)\n")
        f.write("=" * 100 + "\n\n")
        for r in sorted_results:
            f.write(f"\n{r['model']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Macro Avg    - Precision: {r['val_precision_macro']:.4f}, Recall: {r['val_recall_macro']:.4f}, F1: {r['val_f1_macro']:.4f}\n")
            f.write(f"  Weighted Avg - Precision: {r['val_precision_weighted']:.4f}, Recall: {r['val_recall_weighted']:.4f}, F1: {r['val_f1_weighted']:.4f}\n")
            if "cv_f1_mean" in r:
                f.write(f"  CV (5-fold)  - F1: {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}, Accuracy: {r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}\n")

    print(f"Saved: {txt_path}")


def select_best_model(results, model_configs):
    best = max(results, key=lambda r: r["val_f1_macro"])
    print(f"\n" + "=" * 80)
    print(f"BEST MODEL SELECTED")
    print("=" * 80)
    print(f"Model: {best['model']}")
    print(f"Validation Accuracy: {best['val_accuracy']:.4f}")
    print(f"Validation F1 (macro): {best['val_f1_macro']:.4f}")
    print(f"Validation F1 (weighted): {best['val_f1_weighted']:.4f}")
    if "cv_f1_mean" in best:
        print(f"CV F1 (mean): {best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}")

    # Find and save the best model
    for name, model in model_configs:
        if name == best['model']:
            best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
            joblib.dump(model, best_model_path)
            print(f"\nSaved best model: {best_model_path}")
            break

    # Save selection metadata
    meta = {
        "selected_model": best["model"],
        "selection_criterion": "val_f1_macro",
        "val_f1_macro": best["val_f1_macro"],
        "val_accuracy": best["val_accuracy"],
        "val_f1_weighted": best["val_f1_weighted"],
    }
    if "cv_f1_mean" in best:
        meta["cv_f1_mean"] = best["cv_f1_mean"]
        meta["cv_f1_std"] = best["cv_f1_std"]
        
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("Phase 2 — Train Classification Models with Dimensionality Reduction")
    print("=" * 80 + "\n")

    X_train, X_val, y_train, y_val, label_encoder, class_names = load_data()
    model_configs = get_model_configs()

    print(f"Total configurations to train: {len(model_configs)}\n")

    results, confusion_matrices, cv_scores, trained_models = train_and_evaluate(
        model_configs, X_train, X_val, y_train, y_val, class_names
    )

    save_results(results, confusion_matrices, class_names)
    
    # Combine model_configs with trained models for select_best_model
    trained_model_configs = [
        (name, trained_models.get(name, model)) 
        for name, model in model_configs
    ]
    best = select_best_model(results, trained_model_configs)

    print("\nDone! Next: run evaluate_model.py for final test set evaluation.")


if __name__ == "__main__":
    main()
