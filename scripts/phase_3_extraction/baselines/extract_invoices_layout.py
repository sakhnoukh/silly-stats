"""
Phase 3 — Layout-aware invoice extraction using OCR boxes

Uses OCR coordinates to build text lines and layout zones:
- header zone
- body zone
- totals zone

Outputs:
  results/invoice_extractions_layout.csv
  results/invoice_extractions_layout.json
  results/extraction_coverage_layout.txt

Usage:
  python scripts/extract_invoices_layout.py
"""

import os
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.append(str(BASELINES_DIR))

import re
import json
import csv
from datetime import datetime
import pandas as pd

import extract_invoices as base

PROJ_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJ_ROOT / "data"
BOX_DIR = DATA_DIR / "ocr_boxes"
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "layout"

FIELDS = base.FIELDS
EXTRA_FIELDS = [
    "payment_terms",
    "document_confidence",
    "invoice_like_score",
    "ocr_quality_score",
    "layout_confidence",
]


# ---------------------------------------------------------------------------
# OCR box helpers
# ---------------------------------------------------------------------------

def load_boxes_for_file(file_name: str):
    path = BOX_DIR / (Path(file_name).stem + ".csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def build_lines_from_boxes(box_df: pd.DataFrame):
    """
    Group by block/paragraph/line and rebuild line text with coordinates.
    """
    line_rows = []
    grouped = box_df.groupby(["block_num", "par_num", "line_num"], sort=False)

    for (block_num, par_num, line_num), g in grouped:
        g = g.sort_values(["left", "word_num"])
        text = " ".join(str(x) for x in g["text"].tolist()).strip()
        if not text:
            continue

        left = int(g["left"].min())
        top = int(g["top"].min())
        right = int((g["left"] + g["width"]).max())
        bottom = int((g["top"] + g["height"]).max())
        width = right - left
        height = bottom - top
        conf = float(g["conf"].mean()) if "conf" in g.columns else -1.0

        line_rows.append({
            "text": text,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": width,
            "height": height,
            "conf": conf,
            "block_num": block_num,
            "par_num": par_num,
            "line_num": line_num,
        })

    lines = pd.DataFrame(line_rows)
    if not lines.empty:
        lines = lines.sort_values(["top", "left"]).reset_index(drop=True)
    return lines


def add_relative_positions(lines: pd.DataFrame):
    if lines.empty:
        return lines

    page_w = max(lines["right"].max(), 1)
    page_h = max(lines["bottom"].max(), 1)

    lines = lines.copy()
    lines["x_rel"] = lines["left"] / page_w
    lines["y_rel"] = lines["top"] / page_h
    lines["is_header"] = lines["y_rel"] < 0.25
    lines["is_middle"] = (lines["y_rel"] >= 0.25) & (lines["y_rel"] < 0.70)
    lines["is_bottom"] = lines["y_rel"] >= 0.70
    lines["is_left_half"] = lines["x_rel"] < 0.5
    lines["is_right_half"] = lines["x_rel"] >= 0.5
    return lines


# ---------------------------------------------------------------------------
# Layout-aware extraction helpers
# ---------------------------------------------------------------------------

def choose_best_line(candidates, score_fn):
    best = None
    best_score = -999
    for _, row in candidates.iterrows():
        score = score_fn(row)
        if score > best_score:
            best_score = score
            best = row
    return best


def extract_issuer_from_layout(lines):
    if lines is None or lines.empty:
        return None

    header = lines[lines["is_header"]].copy()
    if header.empty:
        return None

    def score(row):
        text = str(row["text"])
        s = 0
        if base._is_bad_name_candidate(text):
            s -= 10
        if re.search(r"\b(inc\.?|corp\.?|ltd\.?|company|corporation|group|institute|research|center|services?|laboratories?)\b", text, re.IGNORECASE):
            s += 5
        if row["y_rel"] < 0.12:
            s += 3
        if row["is_left_half"]:
            s += 2
        if sum(ch.isdigit() for ch in text) == 0:
            s += 1
        if 2 <= len(text.split()) <= 8:
            s += 2
        return s

    best = choose_best_line(header, score)
    if best is None:
        return None
    
    cleaned = base._clean_name(str(best["text"]))
    if not cleaned:
        return None

    if re.search(r"\b(pay to|reference|explanation|client copy|check request)\b", cleaned, re.IGNORECASE):
        return None

    if sum(ch.isdigit() for ch in cleaned) > 5:
        return None

    return cleaned


def extract_recipient_from_layout(lines, issuer_name=None):
    if lines is None or lines.empty:
        return None

    cands = lines[(lines["is_header"] | lines["is_middle"]) & lines["is_left_half"]].copy()
    if cands.empty:
        return None

    def score(row):
        text = str(row["text"])
        s = 0
        lower = text.lower()

        if issuer_name and issuer_name.lower() == text.lower():
            s -= 10
        if re.search(r"\b(bill to|sold to|ship to|to:)\b", lower):
            s += 4
        if re.search(r"\b(attention|attn)\b", lower):
            s += 2
        if re.search(r"\b(inc\.?|corp\.?|ltd\.?|company|corporation|group|institute|research|center|services?|laboratories?)\b", text, re.IGNORECASE):
            s += 3
        if re.search(r"\b(street|avenue|road|suite|floor|telephone|fax|telex)\b", lower):
            s -= 3
        if re.search(r"\b(remit to|reference|explanation|client copy)\b", lower):
            s -= 5
        if sum(ch.isdigit() for ch in text) > 6:
            s -= 3
        if 2 <= len(text.split()) <= 8:
            s += 2
        if re.search(r"\b(pay to|remit to|reference|explanation|client copy|check request)\b", lower):
            s -= 7
        if re.search(r"\b(inc\.?|corp\.?|ltd\.?|company|corporation|group|institute|research|center|services?|laboratories?)\b", text, re.IGNORECASE):
            s += 4
        if re.search(r"^(mr|mrs|ms|dr)\.?\b", text, re.IGNORECASE):
            s += 1
        return s

    best = choose_best_line(cands, score)
    if best is None:
        return None

    text = str(best["text"])
    text = re.sub(r"^(attention|attn)\s*:\s*", "", text, flags=re.IGNORECASE)
    cleaned = base._clean_name(text)

    lower_clean = cleaned.lower()

    if issuer_name and cleaned.lower() == issuer_name.lower():
        return None

    bad_patterns = [
        r"\b(remit to|reference|explanation|client copy|check request)\b",
        r"\b(street|avenue|road|suite|floor|telephone|fax|telex)\b",
        r"^\s*(pay to)\b",
    ]

    for pat in bad_patterns:
        if re.search(pat, lower_clean, re.IGNORECASE):
            return None

    if sum(ch.isdigit() for ch in cleaned) > 5:
        return None

    if not cleaned:
        return None
    return cleaned


def extract_total_from_layout(lines):
    if lines is None or lines.empty:
        return None

    cands = lines[lines["is_bottom"] | lines["is_right_half"]].copy()
    if cands.empty:
        cands = lines.copy()

    amount_re = re.compile(r"\$?\s*([\d,]+(?:\.\d{2})?)")

    best_amt = None
    best_score = -999

    for _, row in cands.iterrows():
        text = str(row["text"])
        lower = text.lower()

        for m in amount_re.finditer(text):
            amt = base._normalise_amount(m.group(1))
            if not amt:
                continue

            try:
                amt_value = float(amt)
            except Exception:
                continue

            # reject implausible amounts unless on a very strong total line
            if amt_value < 10:
                continue

            score = 0
            strong_total = bool(re.search(r"\b(total amount due|amount due|balance due|net amount|this bill total)\b", lower))
            weak_total = bool(re.search(r"\btotal\b", lower))

            if strong_total:
                score += 10
            elif weak_total:
                score += 5
            else:
                score -= 2

            if row["is_bottom"]:
                score += 2
            if row["is_right_half"]:
                score += 2

            # reject weird huge OCR numbers unless strongly supported
            if amt_value > 100000 and not strong_total:
                score -= 6

            # dotted leader lines can be good, but random OCR-heavy lines are bad
            if text.count(".") > 8:
                score += 1

            if score > best_score:
                best_score = score
                best_amt = amt

    return best_amt


def extract_invoice_number_from_layout(lines):
    if lines is None or lines.empty:
        return None

    header = lines[lines["is_header"]].copy()
    if header.empty:
        header = lines.copy()

    patterns = [
        r"(?:invoice|invorce|lnvoice)\s*(?:no|number|#)?\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\breference\s*(?:no|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bour\s*file\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bdoc(?:ument)?\s*(?:no|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
    ]

    for _, row in header.iterrows():
        text = str(row["text"])
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if base._valid_invoice_number(val):
                    return val
    return None


def extract_invoice_date_from_layout(lines):
    if lines is None or lines.empty:
        return None

    header = lines[lines["is_header"]].copy()
    if header.empty:
        header = lines.copy()

    for _, row in header.iterrows():
        text = str(row["text"])
        if re.search(r"\b(date|invoice date|dated)\b", text, re.IGNORECASE):
            m = re.search(
                r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
                text,
                re.IGNORECASE,
            )
            if m:
                norm = base._normalise_date(m.group(1))
                if norm:
                    return norm

    return None


def extract_due_date_from_layout(lines):
    if lines is None or lines.empty:
        return None

    for _, row in lines.iterrows():
        text = str(row["text"])
        if re.search(r"\b(due date|payment due|due)\b", text, re.IGNORECASE):
            m = re.search(
                r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
                text,
                re.IGNORECASE,
            )
            if m:
                norm = base._normalise_date(m.group(1))
                if norm:
                    return norm
    return None


def extract_payment_terms_from_layout(lines):
    if lines is None or lines.empty:
        return None

    for _, row in lines.iterrows():
        text = str(row["text"])
        m = re.search(
            r"\b(net\s*\d{1,3}(?:\s*days?)?|due\s+upon\s+receipt|cash\s+\d+\s*days?|cod)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def extract_layout_fields(file_name, raw_text):
    result = {
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "issuer_name": None,
        "recipient_name": None,
        "total_amount": None,
        "payment_terms": None,
        "document_confidence": "low",
        "invoice_like_score": base.compute_invoice_like_score(raw_text),
        "ocr_quality_score": base.compute_ocr_quality_score(raw_text),
        "layout_confidence": "none",
    }

    if result["invoice_like_score"] <= 0:
        result["layout_confidence"] = "low"
        result["document_confidence"] = "low"
        return result

    box_df = load_boxes_for_file(file_name)
    if box_df is None or box_df.empty:
        return result

    lines = build_lines_from_boxes(box_df)
    if lines is None or lines.empty:
        return result

    lines = add_relative_positions(lines)

    result["issuer_name"] = extract_issuer_from_layout(lines)
    result["recipient_name"] = extract_recipient_from_layout(lines, issuer_name=result["issuer_name"])
    result["invoice_number"] = extract_invoice_number_from_layout(lines)
    result["invoice_date"] = extract_invoice_date_from_layout(lines)
    result["due_date"] = extract_due_date_from_layout(lines)
    result["payment_terms"] = extract_payment_terms_from_layout(lines)
    result["total_amount"] = extract_total_from_layout(lines)

    base_result = base.extract_all_fields(raw_text)

    if not result["invoice_number"]:
        result["invoice_number"] = base_result.get("invoice_number")
    if not result["invoice_date"]:
        result["invoice_date"] = base_result.get("invoice_date")
    if not result["due_date"]:
        result["due_date"] = base_result.get("due_date")
    if not result["payment_terms"]:
        result["payment_terms"] = base_result.get("payment_terms")

    found = sum(1 for k in FIELDS if result.get(k))
    if found >= 4:
        result["layout_confidence"] = "high"
        result["document_confidence"] = "high"
    elif found >= 2:
        result["layout_confidence"] = "medium"
        result["document_confidence"] = "medium"
    else:
        result["layout_confidence"] = "low"
        result["document_confidence"] = "low"

    result = base.validate_extractions(result)
    return result


# ---------------------------------------------------------------------------
# Save/report
# ---------------------------------------------------------------------------

def save_csv(results, invoices_df, path):
    rows = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        rows.append({
            "doc_id": row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split": row.get("_split", ""),
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
            "doc_id": row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split": row.get("_split", ""),
            "fields": extraction,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def save_coverage_report(coverage, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Phase 3 — Layout Extraction Coverage Report\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"  {'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}\n")
        f.write("  " + "-" * 50 + "\n")
        for field, stats in coverage.items():
            f.write(
                f"  {field:<20} {stats['found']:>6}  {stats['total']:>6}  {stats['pct']:>7.1f}%\n"
            )
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3 — Layout-aware Invoice Extraction")
    print("=" * 60)

    invoices = base.load_invoices()

    print("\nExtracting fields with OCR-layout features...")
    results = [
        extract_layout_fields(row["file_name"], row["raw_text"])
        for _, row in invoices.iterrows()
    ]

    coverage = base.compute_coverage(results)
    base.print_coverage_report(coverage)
    base.print_spot_check(results, invoices, n=5)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "invoice_extractions_layout.csv"
    json_path = RESULTS_DIR / "invoice_extractions_layout.json"
    cov_path = RESULTS_DIR / "extraction_coverage_layout.txt"

    save_csv(results, invoices, csv_path)
    save_json(results, invoices, json_path)
    save_coverage_report(coverage, cov_path)

    print(f"\n✓ {csv_path}")
    print(f"✓ {json_path}")
    print(f"✓ {cov_path}")
    print("\nDone. Compare layout-aware results against baseline/template/ML.")


if __name__ == "__main__":
    main()