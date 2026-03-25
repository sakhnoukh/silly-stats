"""
Create smaller candidate-labeling CSVs for manual annotation.

Outputs:
  - results/candidates_recipient_name_labeled.csv
  - results/candidates_invoice_number_labeled.csv
  - results/candidates_total_amount_labeled.csv

It copies candidate tables and adds:
  - label
  - notes

Usage:
    python scripts/make_candidate_labeling_samples.py
"""

from pathlib import Path
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"

FILES = [
    ("candidates_recipient_name.csv", "candidates_recipient_name_labeled.csv"),
    ("candidates_invoice_number.csv", "candidates_invoice_number_labeled.csv"),
    ("candidates_total_amount.csv", "candidates_total_amount_labeled.csv"),
]


def main():
    print("=" * 60)
    print("Create Candidate Labeling Files")
    print("=" * 60)

    for src_name, dst_name in FILES:
        src = RESULTS_DIR / src_name
        dst = RESULTS_DIR / dst_name

        df = pd.read_csv(src)

        if "label" not in df.columns:
            df["label"] = ""
        if "notes" not in df.columns:
            df["notes"] = ""

        df.to_csv(dst, index=False, encoding="utf-8")
        print(f"✓ {dst}")

    print("\nOpen the *_labeled.csv files and set:")
    print("  label = 1 for the correct candidate for that doc/field")
    print("  label = 0 for incorrect candidates")
    print("\nFor docs with no correct candidate present, label all rows 0.")


if __name__ == "__main__":
    main()