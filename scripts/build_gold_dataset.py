"""
Build Gold Evaluation Dataset from Labeled Candidate Tables

Derives ground-truth field values for each document by reading the
labeled candidate CSVs produced by the ML labeling pipeline.

For each (doc_id, field), the gold value is the candidate_text where
label=1.  If no candidate was labeled positive, the gold value is None
(field absent or not labeled in that document).

Covered fields   : recipient_name, invoice_number, total_amount
Source files     : results/phase3_extraction/candidates/*_labeled.csv
Output           : results/gold_dataset.csv

Usage:
    python scripts/build_gold_dataset.py
"""

from pathlib import Path
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
CANDS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"
OUT_PATH  = PROJ_ROOT / "results" / "gold_dataset.csv"

FIELD_FILES = {
    "recipient_name":  CANDS_DIR / "candidates_recipient_name_labeled.csv",
    "invoice_number":  CANDS_DIR / "candidates_invoice_number_labeled.csv",
    "total_amount":    CANDS_DIR / "candidates_total_amount_labeled.csv",
}

FIELDS = list(FIELD_FILES.keys())


def _gold_from_candidates(csv_path: Path, field: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns [doc_id, file_name, _split, <field>]
    where <field> is the first positive candidate text, or None if absent.
    """
    df = pd.read_csv(csv_path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # All unique docs in this labeled set
    doc_meta = (
        df[["doc_id", "file_name", "_split"]]
        .drop_duplicates(subset=["doc_id"])
        .set_index("doc_id")
    )

    # First positive per doc
    positives = (
        df[df["label"] == 1]
        .groupby("doc_id")["candidate_text"]
        .first()
        .rename(field)
    )

    merged = doc_meta.join(positives, how="left")
    merged.index.name = "doc_id"
    return merged.reset_index()


def main():
    frames = []
    for field, csv_path in FIELD_FILES.items():
        print(f"Loading {field} from {csv_path.name} ...", end="  ")
        df = _gold_from_candidates(csv_path, field)
        covered = df[field].notna().sum()
        print(f"{len(df)} docs,  {covered} with gold value")
        frames.append(df.set_index("doc_id"))

    # Rebuild with explicit outer merge
    all_docs = sorted(set().union(*[set(f.index) for f in frames]))
    result = pd.DataFrame({"doc_id": all_docs})

    for field, csv_path in FIELD_FILES.items():
        df = _gold_from_candidates(csv_path, field)
        result = result.merge(
            df[["doc_id", "file_name", "_split", field]],
            on="doc_id",
            how="left",
            suffixes=("", f"_{field}"),
        )

    # Consolidate file_name and _split (multiple merges may create duplicates)
    for col in ["file_name", "_split"]:
        cols = [c for c in result.columns if c == col or c.startswith(f"{col}_")]
        result[col] = result[cols].bfill(axis=1).iloc[:, 0]
        extras = [c for c in cols if c != col]
        result.drop(columns=extras, inplace=True)

    # Final column order
    result = result[["doc_id", "file_name", "_split"] + FIELDS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\nGold dataset: {len(result)} documents")
    print(f"Fields covered:")
    for field in FIELDS:
        n = result[field].notna().sum()
        print(f"  {field:20s}: {n}/{len(result)} docs with gold value ({100*n/len(result):.0f}%)")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
