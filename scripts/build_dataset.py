"""
Build Final Labeled Dataset + Train / Validation / Test Splits
Input:  data/processed/dataset_clean.csv
Output: data/processed/dataset.csv, train.csv, val.csv, test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJ_ROOT, "data", "processed", "dataset_clean.csv")
PROCESSED_DIR = os.path.join(PROJ_ROOT, "data", "processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


def run_build():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} documents from {INPUT_CSV}")

    # -----------------------------------------------------------------------
    # Quality checks
    # -----------------------------------------------------------------------
    # Drop rows with empty clean_text
    empty_mask = df["clean_text"].isna() | (df["clean_text"].str.strip() == "")
    n_empty = empty_mask.sum()
    if n_empty > 0:
        print(f"  WARNING: dropping {n_empty} documents with empty clean_text")
        df = df[~empty_mask].reset_index(drop=True)

    # Check for duplicate texts
    n_dup = df.duplicated(subset=["clean_text"]).sum()
    if n_dup > 0:
        print(f"  WARNING: {n_dup} duplicate clean_text entries found (keeping first)")
        df = df.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)

    # Add helper columns
    df["text_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()

    # Class balance report
    print("\nClass distribution:")
    for label, group in df.groupby("label"):
        print(f"  [{label}] {len(group)} docs | avg words: {group['word_count'].mean():.0f}")

    # Save final dataset
    dataset_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"\nFinal dataset saved: {dataset_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Stratified split: train / val / test
    # -----------------------------------------------------------------------
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"], random_state=RANDOM_STATE
    )
    # Second split: val vs test (equal halves of remaining)
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=RANDOM_STATE
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits:")
    print(f"  Train: {len(train_df)} docs -> {train_path}")
    print(f"  Val:   {len(val_df)} docs -> {val_path}")
    print(f"  Test:  {len(test_df)} docs -> {test_path}")

    # Per-split class distribution
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts().to_dict()
        print(f"  {name}: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    run_build()
