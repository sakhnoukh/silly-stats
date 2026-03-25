from pathlib import Path
import re
import pandas as pd
import os

PROJ_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"
LABELING_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "labeling"

SOURCE_FILES = {
    "recipient_name": CANDIDATES_DIR / "candidates_recipient_name_labeled.csv",
    "invoice_number": CANDIDATES_DIR / "candidates_invoice_number_labeled.csv",
    "total_amount": CANDIDATES_DIR / "candidates_total_amount_labeled.csv",
}

RESPONSES_DIR = LABELING_DIR / "label_responses"

LINE_RE = re.compile(
    r"doc_id=(.*?)\s+\|\s+field=(.*?)\s+\|\s+candidate_text=(.*?)\s+\|\s+label=([01])\s*$"
)


def load_responses():
    rows = []
    if not RESPONSES_DIR.is_dir():
        return rows

    for fname in os.listdir(RESPONSES_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(RESPONSES_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                m = LINE_RE.match(line)
                if m:
                    rows.append({
                        "doc_id": m.group(1).strip(),
                        "field": m.group(2).strip(),
                        "candidate_text": m.group(3).strip(),
                        "label": int(m.group(4)),
                    })
    return rows


def main():
    responses = load_responses()
    if not responses:
        print("No parsed response lines found.")
        return

    resp_df = pd.DataFrame(responses).drop_duplicates(
        subset=["doc_id", "field", "candidate_text"],
        keep="last"
    )

    for field, csv_path in SOURCE_FILES.items():
        if not csv_path.exists():
            print(f"Missing labeled source file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            df["label"] = ""

        field_resp = resp_df[resp_df["field"] == field].copy()
        if field_resp.empty:
            continue

        merge_cols = ["doc_id", "field", "candidate_text"]
        df["doc_id"] = df["doc_id"].astype(str)
        field_resp["doc_id"] = field_resp["doc_id"].astype(str)

        merged = df.merge(
            field_resp[merge_cols + ["label"]],
            on=merge_cols,
            how="left",
            suffixes=("", "_new")
        )

        merged["label"] = merged["label_new"].combine_first(merged["label"])
        merged = merged.drop(columns=["label_new"])

        merged.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✓ Updated {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()