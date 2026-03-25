"""
Export grouped candidate-labeling batches for copy/paste into ChatGPT.

For each field, this script groups rows by doc_id and formats:
- doc_id
- file_name
- field
- raw_text excerpt
- all candidates for that doc/field

Outputs:
  - results/labeling_batches_recipient_name.txt
  - results/labeling_batches_invoice_number.txt
  - results/labeling_batches_total_amount.txt

Usage:
    python scripts/export_labeling_batches.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

from pathlib import Path
from scripts.phase_3_extraction.baselines import extract_invoices as base

PROJ_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"
LABELING_RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "labeling"

FIELD_FILES = {
    "recipient_name": CANDIDATES_DIR / "candidates_recipient_name.csv",
    "invoice_number": CANDIDATES_DIR / "candidates_invoice_number.csv",
    "total_amount": CANDIDATES_DIR / "candidates_total_amount.csv",
}

OUTPUT_FILES = {
    "recipient_name": LABELING_RESULTS_DIR / "labeling_batches_recipient_name.txt",
    "invoice_number": LABELING_RESULTS_DIR / "labeling_batches_invoice_number.txt",
    "total_amount": LABELING_RESULTS_DIR / "labeling_batches_total_amount.txt",
}

RAW_TEXT_CHARS = 1200
MAX_GROUPS_PER_FILE = None   # set to e.g. 40 if you want fewer groups


def safe_text(x):
    return "" if pd.isna(x) else str(x)


def build_raw_text_lookup():
    invoices = base.load_invoices().copy()
    invoices["raw_text"] = invoices["raw_text"].fillna("")
    lookup = {}
    for _, row in invoices.iterrows():
        lookup[str(row["doc_id"])] = {
            "file_name": row.get("file_name", ""),
            "_split": row.get("_split", ""),
            "raw_text": row.get("raw_text", ""),
        }
    return lookup


def format_group(doc_id, field, group_df, raw_lookup):
    info = raw_lookup.get(str(doc_id), {})
    file_name = info.get("file_name", "")
    raw_text = safe_text(info.get("raw_text", ""))[:RAW_TEXT_CHARS]

    lines = []
    lines.append("=" * 80)
    lines.append(f"doc_id={doc_id}")
    lines.append(f"file_name={file_name}")
    lines.append(f"field={field}")
    lines.append("")
    lines.append("raw_text_excerpt:")
    lines.append(raw_text if raw_text.strip() else "[EMPTY]")
    lines.append("")
    lines.append("candidates:")

    group_df = group_df.sort_values(["line_idx", "candidate_source", "candidate_text"]).reset_index(drop=True)

    for i, (_, row) in enumerate(group_df.iterrows(), start=1):
        cand = safe_text(row.get("candidate_text", ""))
        source = safe_text(row.get("candidate_source", ""))
        line_idx = row.get("line_idx", "")
        lines.append(
            f"{i}) candidate_text={cand} | source={source} | line_idx={line_idx}"
        )

    lines.append("")
    return "\n".join(lines)


def export_field_batches(field, csv_path, out_path, raw_lookup):
    df = pd.read_csv(csv_path)
    df["doc_id"] = df["doc_id"].astype(str)

    groups = list(df.groupby("doc_id", sort=False))
    if MAX_GROUPS_PER_FILE is not None:
        groups = groups[:MAX_GROUPS_PER_FILE]

    chunks = []
    for doc_id, group_df in groups:
        chunks.append(format_group(doc_id, field, group_df, raw_lookup))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

    print(f"✓ {out_path} ({len(groups)} grouped cases)")


def main():
    print("=" * 60)
    print("Export Labeling Batches")
    print("=" * 60)

    raw_lookup = build_raw_text_lookup()
    LABELING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for field, csv_path in FIELD_FILES.items():
        export_field_batches(field, csv_path, OUTPUT_FILES[field], raw_lookup)

    print("\nDone. Open the .txt files and paste grouped cases into ChatGPT.")


if __name__ == "__main__":
    main()