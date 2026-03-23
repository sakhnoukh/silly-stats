"""
Phase 3 — Invoice Information Extraction

Loads all invoice documents from the processed splits, runs field-level
extraction on each one using regex + rule-based patterns, and produces:
  - results/invoice_extractions.csv     one row per invoice, one col per field
  - results/invoice_extractions.json    same data, structured
  - results/extraction_coverage.txt     per-field coverage report

⚠️  Works on raw_text (not clean_text) — cleaning lowercases and strips
    stopwords, which destroys label keywords like "Invoice No:" or "Bill To:".

Usage:
    python3 scripts/extract_invoices.py

Dependencies:
    pip install python-dateutil   (flexible date parsing)
"""

import os
import re
import json
import csv
from datetime import datetime

import pandas as pd
from dateutil import parser as dateparser

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(PROJ_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


# ============================================================================
# SECTION 1 — FIELD EXTRACTORS
# Each function receives the raw text of one invoice and returns a string
# with the extracted value, or None if nothing was found.
# ============================================================================

# ---------------------------------------------------------------------------
# 1.1 Invoice Number
# ---------------------------------------------------------------------------
# TODO (after exploration): fill in the patterns list based on what you
# actually see in the data. Order = most specific → most general.
# Examples of what to look for: "Invoice No:", "Inv #", "Invoice Number",
# "Bill No:", a bare alphanumeric code near the word "invoice".

INVOICE_NUMBER_PATTERNS = [
    # TODO: add patterns here after running explore_invoices.py
    # Examples (uncomment and adjust):
    # r"[Ii]nvoice\s*[Nn]o\.?\s*[:\-#]?\s*([A-Z0-9\-]{3,20})",
    # r"[Ii]nv\.?\s*#?\s*[:\-]?\s*([A-Z0-9\-]{3,20})",
    # r"[Bb]ill\s*[Nn]o\.?\s*[:\-]?\s*([A-Z0-9\-]{3,20})",
]

def extract_invoice_number(text: str) -> str | None:
    """
    Extract invoice number using keyword-anchored regex patterns.
    Falls back to a broad alphanumeric pattern near the word 'invoice'.
    """
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # TODO: broad fallback — e.g. any INV-XXXX or #XXXX near invoice keyword
    # Uncomment after seeing real data:
    # fallback = re.search(r"(?i)invoice\D{0,30}([A-Z]{0,3}\d{4,10})", text)
    # if fallback:
    #     return fallback.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# 1.2 Invoice Date
# ---------------------------------------------------------------------------
# TODO: fill patterns after exploration. Key question: do dates appear after
# "Invoice Date:", "Date:", "Dated:", or with no label at all?
# Also check what formats appear: MM/DD/YY, Month DD YYYY, YYYY-MM-DD?

INVOICE_DATE_PATTERNS = [
    # TODO: label-anchored patterns go here
    # Examples:
    # r"[Ii]nvoice\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    # r"[Ii]nvoice\s*[Dd]ate\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    # r"\b[Dd]ate\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
]

def extract_invoice_date(text: str) -> str | None:
    """
    Extract invoice date. Tries label-anchored patterns first, then a broad
    date scan. Uses dateutil to normalise whatever format is found.
    """
    for pattern in INVOICE_DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalise_date(match.group(1).strip())

    # TODO: broad fallback — find ANY date-like string in the document
    # Uncomment and adjust after seeing real formats:
    # broad = re.search(
    #     r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
    #     r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b",
    #     text
    # )
    # if broad:
    #     return _normalise_date(broad.group(1).strip())

    return None


# ---------------------------------------------------------------------------
# 1.3 Due Date
# ---------------------------------------------------------------------------
# TODO: check whether due dates appear explicitly ("Due Date: ...", "Pay by ...")
# or only implicitly as payment terms ("Net 30", "Net 60").
# Both cases should be handled separately — a "Net 30" is not a date string.

DUE_DATE_PATTERNS = [
    # TODO:
    # r"[Dd]ue\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    # r"[Pp]ayment\s*[Dd]ue\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    # r"[Pp]ay\s*[Bb]y\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    # r"[Dd]ue\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
]

def extract_due_date(text: str) -> str | None:
    """
    Extract due date. Also handles payment terms like 'Net 30' when no
    explicit date is present (returns the term string, not a date).
    """
    for pattern in DUE_DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalise_date(match.group(1).strip())

    # TODO: payment terms fallback — "Net 30", "Net 60", etc.
    # terms = re.search(r"\b(Net\s*\d{2,3}|COD|Due\s+on\s+receipt)\b", text, re.IGNORECASE)
    # if terms:
    #     return terms.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# 1.4 Issuer Name
# ---------------------------------------------------------------------------
# TODO: this is the hardest field. For these old RVL-CDIP invoices, the issuer
# is usually the company whose letterhead appears at the top. Strategies:
#   (a) first non-empty line of the document (most common for letterhead)
#   (b) line before "To:" or "Bill To:"
#   (c) line containing "Inc.", "Corp.", "Ltd.", "Co." near the top
#   (d) known company name from top discriminative features:
#       "lorillard", "new york", "york" (from error analysis)

def extract_issuer_name(text: str) -> str | None:
    """
    Extract issuer (sender) name.
    Strategy: company name usually appears in the first 3-5 lines as a header.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if not lines:
        return None

    # TODO: try label-anchored first
    # from_match = re.search(r"[Ff]rom\s*[:\-]?\s*(.+)", text)
    # if from_match:
    #     return from_match.group(1).strip()[:80]

    # TODO: look for company indicators in first N lines
    # for line in lines[:5]:
    #     if re.search(r"\b(Inc|Corp|Ltd|Co|LLC|Company|Industries)\b\.?", line, re.IGNORECASE):
    #         return line[:80]

    # TODO: fallback to first non-trivial line (>= 3 words or looks like a name)
    # for line in lines[:3]:
    #     if len(line.split()) >= 2:
    #         return line[:80]

    return None


# ---------------------------------------------------------------------------
# 1.5 Recipient Name
# ---------------------------------------------------------------------------
# TODO: look for "Bill To:", "Sold To:", "To:", "Ship To:", "Attention:" labels.
# The value is usually on the same line or the next line.

RECIPIENT_PATTERNS = [
    # TODO:
    # r"[Bb]ill\s*[Tt]o\s*[:\-]?\s*(.+)",
    # r"[Ss]old\s*[Tt]o\s*[:\-]?\s*(.+)",
    # r"[Ss]hip\s*[Tt]o\s*[:\-]?\s*(.+)",
    # r"[Aa]ttn\.?\s*[:\-]?\s*(.+)",
    # r"^\s*[Tt]o\s*[:\-]\s*(.+)",   # "To: ..." at line start
]

def extract_recipient_name(text: str) -> str | None:
    """
    Extract recipient (customer) name using label-anchored patterns.
    Also tries the line immediately after a 'Bill To' block.
    """
    for pattern in RECIPIENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            # Take only the first line of the match (name is usually one line)
            value = value.splitlines()[0].strip() if value else value
            if value:
                return value[:80]

    # TODO: multiline block strategy — find "Bill To:" then grab next non-empty line
    # bill_to = re.search(r"[Bb]ill\s*[Tt]o\s*[:\-]?\s*\n\s*(.+)", text)
    # if bill_to:
    #     return bill_to.group(1).strip()[:80]

    return None


# ---------------------------------------------------------------------------
# 1.6 Total Amount
# ---------------------------------------------------------------------------
# TODO: look for "Total", "Total Due", "Amount Due", "Balance Due", "Balance".
# Be careful: there may be multiple amounts (subtotal, tax, total).
# Strategy: find the LAST match — totals usually appear at the bottom.

TOTAL_AMOUNT_PATTERNS = [
    # TODO: label-anchored, ordered specific → general
    # r"[Tt]otal\s*[Aa]mount\s*[Dd]ue\s*[:\-]?\s*([\$£€]?[\d,]+\.?\d*)",
    # r"[Aa]mount\s*[Dd]ue\s*[:\-]?\s*([\$£€]?[\d,]+\.?\d*)",
    # r"[Bb]alance\s*[Dd]ue\s*[:\-]?\s*([\$£€]?[\d,]+\.?\d*)",
    # r"[Tt]otal\s*[Dd]ue\s*[:\-]?\s*([\$£€]?[\d,]+\.?\d*)",
    # r"\b[Tt]otal\s*[:\-]?\s*([\$£€]?[\d,]+\.?\d*)",
]

def extract_total_amount(text: str) -> str | None:
    """
    Extract total amount. Uses label-anchored patterns; returns the LAST match
    found (totals usually appear at the end of the document).
    """
    last_match = None

    for pattern in TOTAL_AMOUNT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            last_match = match.group(1).strip()

    if last_match:
        return _normalise_amount(last_match)

    # TODO: broad fallback — find currency amounts near bottom of doc
    # lines = text.splitlines()
    # for line in reversed(lines[-20:]):  # check last 20 lines
    #     amount = re.search(r"([\$£€]?[\d,]+\.\d{2})", line)
    #     if amount:
    #         return _normalise_amount(amount.group(1).strip())

    return None


# ============================================================================
# SECTION 2 — HELPERS
# ============================================================================

def _normalise_date(date_str: str) -> str | None:
    """
    Try to parse a date string into a consistent YYYY-MM-DD format.
    Returns the original string if parsing fails (better than None).
    """
    if not date_str:
        return None
    try:
        parsed = dateparser.parse(date_str, dayfirst=False)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        # Return raw string — better than discarding it
        return date_str


def _normalise_amount(amount_str: str) -> str | None:
    """
    Normalise a currency string: strip symbols, normalise commas.
    Returns a clean decimal string like "1245.00".
    """
    if not amount_str:
        return None
    cleaned = re.sub(r"[£€\$]", "", amount_str)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip()
    # Validate it's actually a number
    try:
        float(cleaned)
        return cleaned
    except ValueError:
        return amount_str  # return raw if we can't parse


def extract_all_fields(text: str) -> dict:
    """
    Run all 6 extractors on a single invoice text.
    Returns a dict with all fields (None for anything not found).
    """
    return {
        "invoice_number":   extract_invoice_number(text),
        "invoice_date":     extract_invoice_date(text),
        "due_date":         extract_due_date(text),
        "issuer_name":      extract_issuer_name(text),
        "recipient_name":   extract_recipient_name(text),
        "total_amount":     extract_total_amount(text),
    }


# ============================================================================
# SECTION 3 — DATA LOADING
# ============================================================================

def load_invoices() -> pd.DataFrame:
    """
    Load all three splits, concatenate, and keep only invoice-labelled rows.
    Uses raw_text for extraction (not clean_text — cleaning strips keywords).
    """
    dfs = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(PROC_DIR, split)
        if not os.path.exists(path):
            print(f"  ⚠️  {split} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["_split"] = split.replace(".csv", "")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No processed CSVs found in data/processed/. "
            "Run the Phase 1 pipeline first, or ask your teammate for the CSVs."
        )

    all_df = pd.concat(dfs, ignore_index=True)
    invoices = all_df[all_df["label"] == "invoice"].copy()
    invoices["raw_text"] = invoices["raw_text"].fillna("")

    print(f"Loaded {len(invoices)} invoice documents "
          f"({invoices['_split'].value_counts().to_dict()})")

    return invoices


# ============================================================================
# SECTION 4 — EVALUATION / COVERAGE REPORT
# ============================================================================

def compute_coverage(results: list[dict]) -> dict:
    """
    For each field, compute what % of invoices returned a non-null value.
    """
    if not results:
        return {}

    n = len(results)
    coverage = {}
    for field in FIELDS:
        found = sum(1 for r in results if r.get(field) is not None)
        coverage[field] = {"found": found, "total": n, "pct": round(100 * found / n, 1)}
    return coverage


def print_coverage_report(coverage: dict) -> None:
    print("\n── Extraction Coverage ───────────────────────────────────────────────")
    print(f"  {'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}")
    print("  " + "-" * 48)
    for field, stats in coverage.items():
        bar = "█" * int(stats["pct"] / 5)  # visual bar, 1 block per 5%
        print(f"  {field:<20} {stats['found']:>6}  {stats['total']:>6}  "
              f"{stats['pct']:>7.1f}%  {bar}")


def save_coverage_report(coverage: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("Phase 3 — Extraction Coverage Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}\n")
        f.write("-" * 48 + "\n")
        for field, stats in coverage.items():
            f.write(f"{field:<20} {stats['found']:>6}  {stats['total']:>6}  "
                    f"{stats['pct']:>7.1f}%\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# SECTION 5 — SAVE OUTPUTS
# ============================================================================

def save_csv(results: list[dict], invoices_df: pd.DataFrame, path: str) -> None:
    """
    Save extraction results as CSV, one row per invoice.
    Includes doc_id and file_name for traceability.
    """
    rows = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        rows.append({
            "doc_id":    row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split":    row.get("_split", ""),
            **extraction,
        })

    fieldnames = ["doc_id", "file_name", "_split"] + FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(results: list[dict], invoices_df: pd.DataFrame, path: str) -> None:
    output = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        output.append({
            "doc_id":    row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split":    row.get("_split", ""),
            "fields":    extraction,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ============================================================================
# SECTION 6 — MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 3 — Invoice Information Extraction")
    print("=" * 60 + "\n")

    # ── Load data ────────────────────────────────────────────────────────────
    invoices = load_invoices()

    # ── Run extraction ───────────────────────────────────────────────────────
    print("\nExtracting fields...")
    results = []
    for _, row in invoices.iterrows():
        extracted = extract_all_fields(row["raw_text"])
        results.append(extracted)

    # ── Coverage report ──────────────────────────────────────────────────────
    coverage = compute_coverage(results)
    print_coverage_report(coverage)

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path  = os.path.join(RESULTS_DIR, "invoice_extractions.csv")
    json_path = os.path.join(RESULTS_DIR, "invoice_extractions.json")
    cov_path  = os.path.join(RESULTS_DIR, "extraction_coverage.txt")

    save_csv(results, invoices, csv_path)
    save_json(results, invoices, json_path)
    save_coverage_report(coverage, cov_path)

    print(f"\n✓ Saved: {csv_path}")
    print(f"✓ Saved: {json_path}")
    print(f"✓ Saved: {cov_path}")
    print("\nDone. Next: fill in regex patterns in SECTION 1 based on exploration findings.")


if __name__ == "__main__":
    main()