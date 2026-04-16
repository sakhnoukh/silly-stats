"""
Phase 3 — Invoice Information Extraction

Loads all invoice documents from the processed splits, runs field-level
extraction on each one using regex + rule-based patterns, and produces:
  - results/invoice_extractions.csv     one row per invoice, one col per field
  - results/invoice_extractions.json    same data, structured
  - results/extraction_coverage.txt     per-field coverage report

⚠️  Works on raw_text (not clean_text) — cleaning lowercases and strips
    stopwords, which destroys label keywords like "Invoice Date:" or "Attn:".

Usage:
    python scripts/extract_invoices.py

Dependencies (add to requirements.txt if missing):
    python-dateutil>=2.8.2
"""

import os
import re
import json
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from dateutil import parser as dateparser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    print("⚠️  python-dateutil not installed. Dates will be returned as raw strings.")
    print("    Run: pip install python-dateutil\n")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = PROJ_ROOT / "data" / "processed"
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "baseline"

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]

EXTRA_FIELDS = [
    "payment_terms",
    "document_confidence",
    "invoice_like_score",
    "ocr_quality_score",
]


# ============================================================================
# SECTION 1 — FIELD EXTRACTORS
# Patterns ordered: most specific (labelled) → most general (fallback)
# All work on raw_text to preserve original casing and keywords.
# Informed by exploration of 198 RVL-CDIP invoice documents.
# ============================================================================

# ---------------------------------------------------------------------------
# 1.1  Invoice Number
# ---------------------------------------------------------------------------
# From samples: "invoice 67550435", "invorce 90220", "invoice 1558-1"
# Label almost never explicit — number usually follows the word "invoice"
# OCR variants seen: "invorce", "lnvoice", "invoace"

def _valid_invoice_number(val):
    if not val:
        return False

    val = str(val).strip()

    if len(val) < 4 or len(val) > 24:
        return False

    if not re.search(r"\d", val):
        return False

    # reject pure 1-letter prefix + short number unless strong label nearby
    if re.match(r"^[A-Z]\d{3}$", val):
        return False

    if re.match(r"^(amount|date|number|total|page|client|reference|explanation)$", val, re.IGNORECASE):
        return False

    if re.match(r"^[a-z]{5,}$", val):
        return False

    if re.search(r"[^\w\-\/]", val):
        return False

    return True


def extract_invoice_number(text):
    patterns = [
        r"(?:invoice|invorce|lnvoice|invoace)\s*(?:no|number|num|#)?\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\breference\s*(?:no|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bour\s*file\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bdoc(?:ument)?\s*(?:no|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bestimate\s*#\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if _valid_invoice_number(val):
                return val
    return None


# ---------------------------------------------------------------------------
# 1.2  Invoice Date
# ---------------------------------------------------------------------------
# From samples:
#   "invoice date_october 23 1984"  →  Month DD YYYY (42 docs)
#   "10-03-85", "9-21-85"           →  MM-DD-YY (45 docs)
#   "04/01/1993"                    →  MM/DD/YYYY (4 docs)
#   "march 21 1984", "june 24 1998" →  written month

def extract_invoice_date(text):
    patterns = [
        # "invoice date_october 23 1984"  "invoice date: 04/01/1993"
        r"invoice\s*date[_\s:.\-]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"invoice\s*date[_\s:.\-]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        # "dated october 23 1989"
        r"\bdated?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        # "date ___october 23 1989"  (underscores = fill-in blanks)
        r"\bdate\s*[_\s:]{1,10}([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"\bdate\s*[_\s:]{1,10}(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        # Full written month name anywhere
        r"\b((?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},?\s+\d{4})\b",
        # Abbreviated month: "mar 21 1984", "nov 4 1990"
        r"\b([A-Za-z]{3,4}\s+\d{1,2},?\s+\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = _normalise_date(match.group(1).strip())
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# 1.3  Due Date
# ---------------------------------------------------------------------------
# From samples: "net 30 days" (11 docs), "terms net 9-21-85"
# Explicit due dates rare — payment terms more common.

def extract_due_date(text):
    patterns = [
        r"due\s*date\s*[:\-_]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"due\s*date\s*[:\-_]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"pay(?:ment)?\s+(?:due|by)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"terms?\s*[_\s:]{0,5}net\s+(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"\bdue\s+(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalise_date(match.group(1).strip())
    return None


def extract_payment_terms(text):
    patterns = [
        r"\b(net\s*\d{1,3}(?:\s*days?)?)\b",
        r"\b(due\s+upon\s+receipt)\b",
        r"\b(payment\s+is\s+due\s+upon\s+receipt)\b",
        r"\b(cash\s+\d+\s*days?)\b",
        r"\b(COD)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# 1.4  Issuer Name
# ---------------------------------------------------------------------------
# From samples: issuer always in first block as company header.
# "hazleton laboratories america inc", "baker donelson bearman caldwell",
# "borriston laboratories inc.", "metropolitan sunday newspapers inc."

def _is_bad_name_candidate(line):
    bad_patterns = [
        r"\b(invoice|date|amount|total|terms|page|phone|telephone|fax|attn|sold to|bill to|ship to)\b",
        r"\b(remittance|quotation|voucher|check|non-negotiable)\b",
        r"\$",
    ]
    if len(line) < 4 or len(line) > 90:
        return True
    if sum(ch.isdigit() for ch in line) > 6:
        return True
    if re.search(r"\b\d{5,}\b", line):
        return True
    if any(re.search(p, line, re.IGNORECASE) for p in bad_patterns):
        return True
    return False


def extract_issuer_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = lines[:10]

    company_pattern = re.compile(
        r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|co\.?|company|companies|"
        r"laboratories?|labs?|associates?|partners?|group|institute|"
        r"industries|international|consulting|services?|research|"
        r"communications?|advertising|corporation)\b",
        re.IGNORECASE
    )

    best = None
    best_score = -999

    for i, line in enumerate(candidates):
        if _is_bad_name_candidate(line):
            continue

        score = 0
        if company_pattern.search(line):
            score += 4
        if i <= 2:
            score += 2
        if 2 <= len(line.split()) <= 8:
            score += 2
        if sum(ch.isdigit() for ch in line) == 0:
            score += 1
        if line.isupper():
            score += 1
        if re.search(r"[A-Za-z]", line):
            score += 1

        if score > best_score:
            best_score = score
            best = line

    if best_score >= 4:
        return _clean_name(best)
    return None


# ---------------------------------------------------------------------------
# 1.5  Recipient Name
# ---------------------------------------------------------------------------
# From samples: "attn dr. j. daniel heck", "attention melanee b bennett"
# "attn" appears in 23 docs (11.6%) — best signal we have.
# Recipients often Lorillard or Philip Morris in this dataset.

def _extract_block_after_label(text, label_patterns, max_lines=4):
    lines = [l.rstrip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        for lp in label_patterns:
            if re.search(lp, line, re.IGNORECASE):
                block = []
                for j in range(i + 1, min(i + 1 + max_lines, len(lines))):
                    candidate = lines[j].strip()
                    if not candidate:
                        break
                    if re.search(r"\b(invoice|date|terms|amount|total|description|qty|quantity)\b", candidate, re.IGNORECASE):
                        break
                    block.append(candidate)
                if block:
                    return block
    return None

def extract_recipient_name(text):
    block = _extract_block_after_label(
        text,
        [r"\bbill\s*to\b", r"\bsold\s*to\b", r"\bship\s*to\b", r"^to\s*:?$"],
        max_lines=4
    )

    if block:
        for line in block:
            if len(line.split()) >= 2 and sum(c.isdigit() for c in line) <= 4:
                cleaned = re.split(r"\b(p\.?o\.?\s*box|\d{5,})\b", line, flags=re.IGNORECASE)[0].strip()
                if cleaned and len(cleaned) >= 3:
                    return _clean_name(cleaned)

    # fallback: ATTN line
    m = re.search(
        r"att(?:n|ention)\.?\s+(?:dr\.?\s+|mr\.?\s+|ms\.?\s+|mrs\.?\s+)?([A-Za-z][A-Za-z\s\.\-&]{3,60})",
        text,
        re.IGNORECASE
    )
    if m:
        return _clean_name(m.group(1))

    return None

# ---------------------------------------------------------------------------
# 1.6  Total Amount
# ---------------------------------------------------------------------------
# From samples: "total amount due $ 1,350.00", "total amount due $ 123,250.00"
# Also "totaal 50.926" (Belgian), bare decimals.
# No $ symbol in most docs — amounts are bare decimals.
# Returns LAST match — totals at bottom of document.

def extract_total_amount(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []

    amount_re = re.compile(r"\$?\s*([\d,]+(?:\.\d{2})?)")

    for idx, line in enumerate(lines):
        lower = line.lower()

        for m in amount_re.finditer(line):
            raw_amt = m.group(1)

            if "." not in raw_amt and not re.search(
                r"\b(total|amount due|balance due|net amount)\b", lower
            ):
                continue

            amt = _normalise_amount(raw_amt)
            if not amt:
                continue

            score = 0

            if "total amount due" in lower:
                score += 6
            elif "amount due" in lower or "balance due" in lower:
                score += 5
            elif re.search(r"\btotal\b", lower):
                score += 4

            if idx >= int(len(lines) * 0.5):
                score += 1

            if line.count(".") > 8:
                score -= 2

            candidates.append((score, float(amt), amt, line))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if candidates[0][0] >= 2:
            return candidates[0][2]

    return None


# ============================================================================
# SECTION 2 — HELPERS
# ============================================================================

def _normalise_date(date_str):
    if not date_str:
        return None

    date_str = re.sub(r"\s+[a-zA-Z]\s+(?=\d{2,4})", " ", date_str).strip()

    numeric = re.match(r"(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})", date_str)
    if numeric:
        a, b = int(numeric.group(1)), int(numeric.group(2))
        if a > 31 or b > 31:
            return None

    if DATEUTIL_AVAILABLE:
        try:
            parsed = dateparser.parse(date_str, dayfirst=False, fuzzy=False)
            if parsed and 1960 <= parsed.year <= 2005:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    return None


def _normalise_amount(amount_str):
    """Strip currency symbols, return plain decimal string."""
    if not amount_str:
        return None
    cleaned = re.sub(r"[£€\$\s]", "", amount_str).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if val <= 0:
            return None
        return f"{val:.2f}"
    except ValueError:
        return None


def _clean_name(name):
    """Strip leading punctuation, collapse whitespace, truncate."""
    name = re.sub(r"^[\s\W]+", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()[:100]

def compute_ocr_quality_score(text):
    if not text or not str(text).strip():
        return 0.0

    t = str(text)
    chars = len(t)
    if chars == 0:
        return 0.0

    alpha = sum(c.isalpha() for c in t) / chars
    digits = sum(c.isdigit() for c in t) / chars
    spaces = sum(c.isspace() for c in t) / chars

    lines = [l.strip() for l in t.splitlines() if l.strip()]
    avg_line_len = sum(len(l) for l in lines) / len(lines) if lines else 0

    weird = len(re.findall(r"[^A-Za-z0-9\s\.,:/\-&()$#]", t)) / chars

    score = 1.0
    if alpha < 0.45:
        score -= 0.25
    if spaces < 0.08:
        score -= 0.15
    if avg_line_len < 12:
        score -= 0.15
    if weird > 0.08:
        score -= 0.20
    if digits > 0.35:
        score -= 0.10

    return max(0.0, round(score, 2))

def compute_invoice_like_score(text):
    t = text.lower()

    positive = 0
    negative = 0

    positive_terms = [
        r"\binvoice\b",
        r"\binvoice\s*(?:no|number|#)\b",
        r"\binvoice\s*date\b",
        r"\bbill\s*to\b",
        r"\bsold\s*to\b",
        r"\bamount\s+due\b",
        r"\btotal\s+amount\s+due\b",
        r"\bpayment\s+terms\b",
    ]

    negative_terms = [
        r"\bquotation\b",
        r"\bquote\b",
        r"\bremittance advice\b",
        r"\bvoucher\b",
        r"\bcheck request\b",
        r"\bnon-negotiable\b",
        r"\bpay to the order of\b",
    ]

    for p in positive_terms:
        if re.search(p, t):
            positive += 1

    for p in negative_terms:
        if re.search(p, t):
            negative += 1

    return positive - negative

def validate_extractions(fields):
    # invoice number
    if fields.get("invoice_number") and not _valid_invoice_number(fields["invoice_number"]):
        fields["invoice_number"] = None

    # names
    for key in ["issuer_name", "recipient_name"]:
        val = fields.get(key)
        if val:
            if sum(ch.isdigit() for ch in val) > 5:
                fields[key] = None
            elif re.search(
                r"\b(amount|total|invoice|date|terms|advertising|request|requisition|copy|client)\b",
                val,
                re.IGNORECASE,
            ):
                fields[key] = None

    # due date should be ISO date only
    if fields.get("due_date") and not re.match(r"\d{4}-\d{2}-\d{2}$", str(fields["due_date"])):
        fields["due_date"] = None

    # total amount should look numeric
    if fields.get("total_amount") and not re.match(r"^\d+(?:\.\d{2})$", str(fields["total_amount"])):
        fields["total_amount"] = None

    return fields

def validate_extractions(fields):
    # invoice number
    if fields.get("invoice_number") and not _valid_invoice_number(fields["invoice_number"]):
        fields["invoice_number"] = None

    # names
    for key in ["issuer_name", "recipient_name"]:
        val = fields.get(key)
        if val:
            if sum(ch.isdigit() for ch in val) > 5:
                fields[key] = None
            elif re.search(
                r"\b(amount|total|invoice|date|terms|advertising|request|requisition|copy|client)\b",
                val,
                re.IGNORECASE,
            ):
                fields[key] = None

    # due date should be ISO date only
    if fields.get("due_date") and not re.match(r"\d{4}-\d{2}-\d{2}$", str(fields["due_date"])):
        fields["due_date"] = None

    # total amount should look numeric
    if fields.get("total_amount") and not re.match(r"^\d+(?:\.\d{2})$", str(fields["total_amount"])):
        fields["total_amount"] = None

    return fields

def extract_all_fields(text):
    """Run all extractors. Returns dict with required + auxiliary fields."""
    invoice_like_score = compute_invoice_like_score(text)
    ocr_quality_score = compute_ocr_quality_score(text)

    if invoice_like_score >= 2:
        document_confidence = "high"
    elif invoice_like_score == 1:
        document_confidence = "medium"
    else:
        document_confidence = "low"

    result = {
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "issuer_name": None,
        "recipient_name": None,
        "total_amount": None,
        "payment_terms": None,
        "invoice_like_score": invoice_like_score,
        "ocr_quality_score": ocr_quality_score,
        "document_confidence": document_confidence,
    }

    # Abstain on unreadable OCR
    if ocr_quality_score < 0.35:
        return result

    result["invoice_number"] = extract_invoice_number(text)
    result["invoice_date"] = extract_invoice_date(text)
    result["due_date"] = extract_due_date(text)
    result["payment_terms"] = extract_payment_terms(text)
    result["issuer_name"] = extract_issuer_name(text)
    result["recipient_name"] = extract_recipient_name(text)
    result["total_amount"] = extract_total_amount(text)

    result = validate_extractions(result)
    return result

# ============================================================================
# SECTION 3 — DATA LOADING
# ============================================================================

def load_invoices():
    """Load all splits, keep invoice rows, use raw_text for extraction."""
    dfs = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = PROC_DIR / split
        if not path.exists():
            print(f"  ⚠️  {split} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["_split"] = split.replace(".csv", "")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No processed CSVs found in data/processed/.\n"
            "Run: python scripts/extract_real_data.py\n"
            "     python scripts/clean_text.py\n"
            "     python scripts/build_dataset.py"
        )

    all_df = pd.concat(dfs, ignore_index=True)
    invoices = all_df[all_df["label"] == "invoice"].copy()
    invoices["raw_text"] = invoices["raw_text"].fillna("")

    split_counts = invoices["_split"].value_counts().to_dict()
    print(f"Loaded {len(invoices)} invoice documents {split_counts}")
    empty = (invoices["raw_text"].str.strip() == "").sum()
    if empty:
        print(f"  ⚠️  {empty} documents have empty raw_text — will return all None")

    return invoices


# ============================================================================
# SECTION 4 — COVERAGE REPORT
# ============================================================================

def compute_coverage(results):
    if not results:
        return {}
    n = len(results)
    return {
        field: {
            "found": sum(1 for r in results if r.get(field) is not None),
            "total": n,
            "pct":   round(100 * sum(1 for r in results if r.get(field) is not None) / n, 1)
        }
        for field in FIELDS
    }


def print_coverage_report(coverage):
    print("\n── Extraction Coverage ───────────────────────────────────────────────")
    print(f"  {'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}")
    print("  " + "-" * 50)
    for field, stats in coverage.items():
        bar = "█" * int(stats["pct"] / 5)
        print(f"  {field:<20} {stats['found']:>6}  {stats['total']:>6}  "
              f"{stats['pct']:>7.1f}%  {bar}")


def save_coverage_report(coverage, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Phase 3 — Extraction Coverage Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"  {'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}\n")
        f.write("  " + "-" * 50 + "\n")
        for field, stats in coverage.items():
            f.write(f"  {field:<20} {stats['found']:>6}  {stats['total']:>6}  "
                    f"{stats['pct']:>7.1f}%\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nNote: coverage on RVL-CDIP (1980s scanned tobacco docs).\n")
        f.write("Expected higher coverage on modern printed invoices.\n")


# ============================================================================
# SECTION 5 — SAVE OUTPUTS
# ============================================================================

def save_csv(results, invoices_df, path):
    rows = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        rows.append({
            "doc_id":    row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split":    row.get("_split", ""),
            **extraction,
        })
    fieldnames = ["doc_id", "file_name", "_split"] + FIELDS + EXTRA_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(results, invoices_df, path):
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
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# SECTION 6 — SPOT CHECK
# ============================================================================

def print_spot_check(results, invoices_df, n=5):
    """Print n examples with source snippet for manual verification."""
    print(f"\n── Spot Check ({n} examples) ─────────────────────────────────────────")
    # Prefer examples where at least 2 fields were extracted
    rich = [(i, r) for i, r in enumerate(results)
        if sum(1 for k, v in r.items() if k in FIELDS and v) >= 2]
    samples = rich[:n] if rich else list(enumerate(results))[:n]

    for i, extraction in samples:
        row = invoices_df.iloc[i]
        print(f"\n  [{row.get('file_name', i)}]")
        snippet = str(row.get("raw_text", ""))[:200].replace("\n", " ")
        print(f"  TEXT: {snippet}…")
        for field, value in extraction.items():
            mark = "✓" if value else "✗"
            print(f"  {mark} {field:<20}: {value}")


# ============================================================================
# SECTION 7 — MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 3 — Invoice Information Extraction")
    print("=" * 60 + "\n")

    invoices = load_invoices()

    print("\nExtracting fields from raw text...")
    results = [extract_all_fields(row["raw_text"]) for _, row in invoices.iterrows()]

    coverage = compute_coverage(results)
    print_coverage_report(coverage)
    print_spot_check(results, invoices, n=5)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "invoice_extractions.csv"
    json_path = RESULTS_DIR / "invoice_extractions.json"
    cov_path = RESULTS_DIR / "extraction_coverage.txt"

    save_csv(results, invoices, csv_path)
    save_json(results, invoices, json_path)
    save_coverage_report(coverage, cov_path)

    print(f"\n✓ {csv_path}")
    print(f"✓ {json_path}")
    print(f"✓ {cov_path}")
    print("\nDone. Review results/phase_3_extraction/baselines/invoice_extractions.csv to verify quality.")


if __name__ == "__main__":
    main()