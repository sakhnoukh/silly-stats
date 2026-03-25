"""
Text Cleaning and Normalization Pipeline
Input:  data/extracted/dataset_raw.csv
Output: data/processed/dataset_clean.csv
"""

import os
import re
import csv

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJ_ROOT, "data", "extracted", "dataset_raw.csv")
OUTPUT_CSV = os.path.join(PROJ_ROOT, "data", "processed", "dataset_clean.csv")

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Clean raw extracted text while preserving signals useful for classification.
    Keeps digits, currency symbols, and email-like patterns.
    """
    text = raw

    # Normalize unicode whitespace
    text = re.sub(r"[\xa0\u200b\u2028\u2029]", " ", text)

    # Lowercase
    text = text.lower()

    # Replace multiple newlines with single space
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove very long sequences of repeated characters (OCR artifacts)
    text = re.sub(r"(.)\1{4,}", r"\1\1", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords but keep digits and currency-like tokens
    cleaned_tokens = []
    for t in tokens:
        if t in STOP_WORDS:
            continue
        # Keep token if it has at least one alphanumeric char or is a currency symbol
        if re.search(r"[a-z0-9$€£¥]", t):
            cleaned_tokens.append(t)

    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cleaning():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} documents from {INPUT_CSV}")

    out_rows = []
    for row in rows:
        raw = row["raw_text"]
        clean = clean_text(raw)
        out_rows.append({
            "doc_id": row["doc_id"],
            "file_name": row["file_name"],
            "label": row["label"],
            "source_dataset": row["source_dataset"],
            "file_type": row["file_type"],
            "extraction_method": row["extraction_method"],
            "raw_text": raw,
            "clean_text": clean,
            "text_length": len(clean),
            "word_count": len(clean.split()),
        })

    fieldnames = ["doc_id", "file_name", "label", "source_dataset", "file_type",
                  "extraction_method", "raw_text", "clean_text", "text_length", "word_count"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # Summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in out_rows)
    print(f"\nCleaning complete: {len(out_rows)} documents -> {OUTPUT_CSV}")
    for label, cnt in sorted(label_counts.items()):
        avg_len = sum(r["text_length"] for r in out_rows if r["label"] == label) / cnt
        print(f"  [{label}] {cnt} docs, avg clean text length: {avg_len:.0f} chars")


if __name__ == "__main__":
    run_cleaning()
