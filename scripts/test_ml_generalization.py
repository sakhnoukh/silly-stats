"""
Test ML Ranker Generalization on Modern Invoices

Runs the trained ML rankers (recipient_name, invoice_number, total_amount)
against the modern invoice test suite to see how well they generalize.

Pipeline:
  1. Extract text from each invoice PDF (via run.py)
  2. Generate candidates per field (same strategy as original pipeline)
  3. Score candidates with trained rankers
  4. Compare top-1 predictions to ground truth

Usage:
    python scripts/test_ml_generalization.py
    python scripts/test_ml_generalization.py --invoices tests/invoices/ --ground-truth tests/ground_truth.json
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

MODELS_DIR = PROJ_ROOT / "models" / "phase3_extraction"

ML_FIELDS = ["recipient_name", "invoice_number", "total_amount"]

MODEL_PATHS = {
    "recipient_name":  MODELS_DIR / "recipient_name_ranker.pkl",
    "invoice_number":  MODELS_DIR / "invoice_number_ranker.pkl",
    "total_amount":    MODELS_DIR / "total_amount_ranker.pkl",
}


# ---------------------------------------------------------------------------
# Candidate generators
# These mirror the logic that was used to build the original candidate CSVs
# for RVL-CDIP, adapted for clean PDF text.
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower() if s else ""


def generate_amount_candidates(text: str, doc_id: str) -> list[dict]:
    """Find all currency-like values in the document."""
    candidates = []
    lines = text.splitlines()

    amount_re = re.compile(
        r'(?:[\$£€¥])\s*([\d,]+\.?\d*)'   # symbol then number
        r'|([\d,]+\.\d{2})'                 # bare decimal
        r'|([\d]{1,3}(?:,\d{3})+\.?\d*)',   # comma-formatted
        re.IGNORECASE
    )

    for line_no, line in enumerate(lines):
        for m in amount_re.finditer(line):
            raw = m.group(0).strip()
            num_str = re.sub(r'[^\d\.]', '', raw)
            try:
                val = float(num_str)
            except ValueError:
                continue
            if val <= 0 or val > 10_000_000:
                continue

            # Features
            line_lower = line.lower()
            near_total   = int(bool(re.search(r'\b(total|amount due|balance|grand total|payable)\b', line_lower)))
            near_subtotal= int(bool(re.search(r'\bsubtotal\b', line_lower)))
            near_tax     = int(bool(re.search(r'\b(tax|vat|gst)\b', line_lower)))
            near_discount= int(bool(re.search(r'\bdiscount\b', line_lower)))
            is_last_amount = 0  # will be updated below
            n_digits     = len(re.findall(r'\d', raw))
            line_pos     = line_no / max(len(lines), 1)
            has_symbol   = int(bool(re.search(r'[\$£€¥]', raw)))

            candidates.append({
                "doc_id":         doc_id,
                "field":          "total_amount",
                "candidate_text": raw,
                "_val":           val,
                "_line_no":       line_no,
                # Features the ranker was trained on
                "near_total_label":    near_total,
                "near_subtotal_label": near_subtotal,
                "near_tax_label":      near_tax,
                "near_discount_label": near_discount,
                "n_digits":            n_digits,
                "line_position":       round(line_pos, 3),
                "has_currency_symbol": has_symbol,
                "candidate_value":     val,
            })

    # Mark the last amount in the doc (likely to be total)
    if candidates:
        candidates[-1]["is_last_amount"] = 1
        for c in candidates[:-1]:
            c["is_last_amount"] = 0

    return candidates


def generate_invoice_number_candidates(text: str, doc_id: str) -> list[dict]:
    """Find alphanumeric codes that could be invoice numbers."""
    candidates = []
    lines = text.splitlines()

    # Pattern: looks like an invoice number (has digit + letter or dashes)
    code_re = re.compile(r'\b([A-Z]{1,5}[-/]?\d{3,}[\w\-/]*|\d{4,}[\-/]\d+[\-/]?\d*)\b')

    for line_no, line in enumerate(lines):
        line_lower = line.lower()
        near_inv_label = int(bool(re.search(
            r'\b(invoice|inv\.?|factura|proforma|our\s+ref|reference|ref\.?)\b',
            line_lower
        )))

        for m in code_re.finditer(line):
            val = m.group(1).strip()
            if len(val) < 3 or len(val) > 25:
                continue
            if not re.search(r'\d', val):
                continue

            n_digits  = len(re.findall(r'\d', val))
            n_letters = len(re.findall(r'[A-Za-z]', val))
            has_dash  = int('-' in val)
            line_pos  = line_no / max(len(lines), 1)

            candidates.append({
                "doc_id":         doc_id,
                "field":          "invoice_number",
                "candidate_text": val,
                "near_inv_label": near_inv_label,
                "n_digits":       n_digits,
                "n_letters":      n_letters,
                "has_dash":       has_dash,
                "candidate_length": len(val),
                "line_position":  round(line_pos, 3),
            })

    return candidates


def generate_recipient_candidates(text: str, doc_id: str) -> list[dict]:
    """Find lines that could be company/person names as recipient."""
    candidates = []
    lines = text.splitlines()

    bill_to_re = re.compile(
        r'\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|client|customer|'
        r'billed\s*to|pay(?:able)?\s*to|factura\s*a|cliente)\b',
        re.IGNORECASE
    )
    company_re = re.compile(
        r'\b(inc\.?|corp\.?|ltd\.?|llc\.?|llp\.?|gmbh|ag|s\.l\.?|s\.a\.?|'
        r'pty\.?|plc\.?|co\.?|group|holdings?|partners?|associates?)\b',
        re.IGNORECASE
    )

    # Find "Bill To" anchor line
    bill_to_line = -1
    for i, line in enumerate(lines):
        if bill_to_re.search(line):
            bill_to_line = i
            break

    for line_no, line in enumerate(lines):
        line_s = line.strip()
        if not line_s or len(line_s) < 3 or len(line_s) > 100:
            continue

        alpha_ratio = sum(c.isalpha() for c in line_s) / max(len(line_s), 1)
        if alpha_ratio < 0.3:
            continue

        near_bill_to = int(bill_to_line >= 0 and 0 < line_no - bill_to_line <= 3)
        has_company  = int(bool(company_re.search(line_s)))
        n_words      = len(line_s.split())
        line_pos     = line_no / max(len(lines), 1)
        is_after_bill= int(bill_to_line >= 0 and line_no > bill_to_line)

        if near_bill_to or has_company or (3 <= n_words <= 8 and alpha_ratio > 0.6):
            candidates.append({
                "doc_id":          doc_id,
                "field":           "recipient_name",
                "candidate_text":  line_s,
                "near_bill_to":    near_bill_to,
                "has_company_ind": has_company,
                "n_words":         n_words,
                "alpha_ratio":     round(alpha_ratio, 3),
                "line_position":   round(line_pos, 3),
                "is_after_bill_to": is_after_bill,
            })

    return candidates[:30]  # cap at 30 candidates per doc


GENERATORS = {
    "total_amount":   generate_amount_candidates,
    "invoice_number": generate_invoice_number_candidates,
    "recipient_name": generate_recipient_candidates,
}


# ---------------------------------------------------------------------------
# Load ranker + score candidates
# ---------------------------------------------------------------------------

def load_ranker(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def make_feature_dicts(df: pd.DataFrame, numeric_cols, categorical_cols) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        feat = {}
        for col in numeric_cols:
            v = row.get(col)
            feat[col] = 0.0 if pd.isna(v) else float(v)
        for col in categorical_cols:
            v = row.get(col)
            feat[col] = "__MISSING__" if pd.isna(v) else str(v)
        rows.append(feat)
    return rows


def predict_top1(ranker: dict, candidates: list[dict]) -> str | None:
    if not candidates:
        return None

    df = pd.DataFrame(candidates)
    pipeline      = ranker["pipeline"]
    numeric_cols  = [c for c in ranker["numeric_cols"]      if c in df.columns]
    categorical_cols = [c for c in ranker["categorical_cols"] if c in df.columns]

    if not numeric_cols and not categorical_cols:
        return None

    feat_dicts = make_feature_dicts(df, numeric_cols, categorical_cols)
    try:
        probs = pipeline.predict_proba(feat_dicts)[:, 1]
    except Exception:
        try:
            probs = pipeline.decision_function(feat_dicts)
        except Exception:
            return None

    best_idx  = int(pd.Series(probs).idxmax())
    best_prob = probs[best_idx]

    if best_prob >= 0.3:
        return str(candidates[best_idx]["candidate_text"])
    return None


# ---------------------------------------------------------------------------
# Matching (same as evaluate_pipeline.py)
# ---------------------------------------------------------------------------

def _to_float(s):
    if not s: return None
    cleaned = re.sub(r"[^\d\.]", "", str(s))
    try: return float(cleaned) if cleaned else None
    except ValueError: return None

def match_string(gold, pred) -> bool:
    g, p = _clean(gold), _clean(pred)
    return bool(g) and bool(p) and (g in p or p in g)

def match_invoice_number(gold, pred) -> bool:
    g = _clean(gold).lstrip("0")
    p = _clean(pred).lstrip("0")
    return bool(g) and g == p

def match_amount(gold, pred) -> bool:
    g, p = _to_float(gold), _to_float(pred)
    if g is None or p is None: return False
    return abs(g - p) <= max(0.01, 0.01 * abs(g))

MATCH_FN = {
    "invoice_number": match_invoice_number,
    "recipient_name": match_string,
    "total_amount":   match_amount,
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(invoices_dir: Path, ground_truth: dict, rankers: dict) -> dict:
    from run import extract_text

    field_correct = {f: 0 for f in ML_FIELDS}
    field_total   = {f: 0 for f in ML_FIELDS}
    per_file      = {}

    invoice_files = sorted(
        p for p in invoices_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".txt"}
    )

    for file_path in invoice_files:
        name = file_path.name
        gt   = ground_truth.get(name)
        if gt is None:
            alt = re.sub(r'_\d+(\.[^.]+)$', r'\1', name)
            gt  = ground_truth.get(alt)
        if gt is None:
            continue

        print(f"  {name} ...", end=" ", flush=True)
        try:
            raw_text = extract_text(str(file_path))
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        doc_id     = name.replace(".pdf", "")
        file_result = {}
        correct_count = 0

        for field in ML_FIELDS:
            gold_val = gt.get(field)
            if gold_val is None:
                continue

            # Generate candidates
            candidates = GENERATORS[field](raw_text, doc_id)

            # Score with ranker
            ranker   = rankers.get(field)
            pred_val = predict_top1(ranker, candidates) if ranker else None

            is_correct = MATCH_FN[field](gold_val, pred_val)
            field_correct[field] += int(is_correct)
            field_total[field]   += 1
            correct_count        += int(is_correct)

            file_result[field] = {
                "gold":       gold_val,
                "predicted":  pred_val,
                "correct":    is_correct,
                "n_candidates": len(candidates),
            }

        n_gold = sum(1 for f in ML_FIELDS if gt.get(f))
        print(f"{correct_count}/{n_gold} correct (ML ranker)")
        per_file[name] = file_result

    field_summary = {}
    for field in ML_FIELDS:
        t = field_total[field]
        c = field_correct[field]
        field_summary[field] = {
            "correct":  c,
            "total":    t,
            "accuracy": round(c / t, 3) if t else None,
        }

    return {"field_summary": field_summary, "per_file": per_file}


def print_report(results: dict) -> None:
    summary = results["field_summary"]
    print()
    print("  ML Ranker — Generalization Test on Modern Invoices")
    print("  (Only 3 fields: invoice_number, recipient_name, total_amount)")
    print("=" * 60)
    print(f"  {'Field':<25} {'Correct':>7}  {'Total':>5}  {'Accuracy':>9}")
    print("  " + "-" * 52)
    total_c = total_t = 0
    for field, stats in summary.items():
        c, t = stats["correct"], stats["total"]
        acc  = stats["accuracy"]
        bar  = "█" * int((acc or 0) * 10)
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        print(f"  {field:<25} {c:>7}  {t:>5}  {acc_str:>9}  {bar}")
        total_c += c; total_t += t
    print("  " + "-" * 52)
    overall = total_c / total_t if total_t else 0
    print(f"  {'OVERALL (3 fields)':<25} {total_c:>7}  {total_t:>5}  {overall:.1%}")
    print()
    print("  Compare to regex v1 on same 3 fields:")
    print("  (Run evaluate_pipeline.py and check invoice_number,")
    print("   recipient_name, total_amount rows for direct comparison)")


def main():
    parser = argparse.ArgumentParser(
        description="Test ML ranker generalization on modern invoice PDFs"
    )
    parser.add_argument("--invoices",      default="tests/invoices")
    parser.add_argument("--ground-truth",  default="tests/ground_truth.json")
    parser.add_argument("--output",        default=None)
    args = parser.parse_args()

    invoices_dir = PROJ_ROOT / args.invoices
    gt_path      = PROJ_ROOT / args.ground_truth

    for p, label in [(invoices_dir, "invoices dir"), (gt_path, "ground truth")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}"); sys.exit(1)

    # Load rankers
    rankers = {}
    for field, model_path in MODEL_PATHS.items():
        if model_path.exists():
            rankers[field] = load_ranker(model_path)
            print(f"Loaded ranker: {field}")
        else:
            print(f"WARNING: ranker not found for {field}: {model_path}")
    print()

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"Evaluating {len(list(invoices_dir.iterdir()))} invoices...")
    print()

    results = evaluate(invoices_dir, ground_truth, rankers)
    print_report(results)

    if args.output:
        out = PROJ_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
