"""
TF-IDF Feature Generation
Input:  data/processed/train.csv, val.csv, test.csv
Output: data/features/tfidf_vectorizer.pkl, X_train.npz, X_val.npz, X_test.npz,
        y_train.npy, y_val.npy, y_test.npy, label_encoder.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJ_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")


def run_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Fill NaN clean_text
    for df in [train_df, val_df, test_df]:
        df["clean_text"] = df["clean_text"].fillna("")

    # -----------------------------------------------------------------------
    # TF-IDF Vectorizer
    # -----------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    print(f"\nTF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val   shape: {X_val.shape}")
    print(f"X_test  shape: {X_test.shape}")

    # -----------------------------------------------------------------------
    # Label encoding
    # -----------------------------------------------------------------------
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    print(f"\nLabel classes: {list(le.classes_)}")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    sparse.save_npz(os.path.join(FEATURES_DIR, "X_train.npz"), X_train)
    sparse.save_npz(os.path.join(FEATURES_DIR, "X_val.npz"), X_val)
    sparse.save_npz(os.path.join(FEATURES_DIR, "X_test.npz"), X_test)

    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)

    joblib.dump(vectorizer, os.path.join(FEATURES_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(le, os.path.join(FEATURES_DIR, "label_encoder.pkl"))

    print(f"\nFeature artifacts saved to {FEATURES_DIR}/")

    # -----------------------------------------------------------------------
    # Quick sanity check: top features per class
    # -----------------------------------------------------------------------
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop 10 TF-IDF features per class (by mean TF-IDF in training set):")
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_train == cls_idx
        if mask.sum() == 0:
            continue
        mean_tfidf = np.asarray(X_train[mask].mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:10]
        top_feats = [(feature_names[i], round(mean_tfidf[i], 4)) for i in top_indices]
        print(f"  [{cls_name}] {top_feats}")


if __name__ == "__main__":
    run_features()
