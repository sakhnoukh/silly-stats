"""
Pipeline Evaluation — Modern Invoice Test Suite

Runs the full pipeline against a set of test invoices with known ground truth
and reports per-field accuracy. Accepts a specific extractor version so you
can compare v0 (original) against v1 (fixed) side by side.

Usage:
    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v1.py
    python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v0.py --output results/eval_v0.json

Typical workflow:
    # Save original as v0
    cp scripts/extract_invoice_fields.py scripts/extract_invoice_fields_v0.py
    python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v0.py --output results/eval_v0.json

    # Save fixed version as v1
    cp <new_file> scripts/extract_invoice_fields_v1.py
    python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v1.py --output results/eval_v1.json
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


# ---------------------------------------------------------------------------
# Dynamic extractor loading
# ---------------------------------------------------------------------------

def load_extractor(extractor_path: Path):
    """
    Dynamically load an InvoiceExtractor class from any .py file.
    Lets you swap versions without touching scripts/extract_invoice_fields.py.
    """
    spec = importlib.util.spec_from_file_location("_extractor_module", extractor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InvoiceExtractor


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _to_float(s):
    if not s:
        return None
    cleaned = re.sub(r"[^\d\.]", "", str(s))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _normalise_unicode(s: str) -> str:
    """Normalize accented characters for fuzzy matching (ñ→n, é→e, etc.)."""
    import unicodedata
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii')


def match_string(gold, pred) -> bool:
    g, p = _clean(gold), _clean(pred)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    # Try again after removing accents (handles ñ vs n, é vs e etc.)
    gn, pn = _normalise_unicode(g), _normalise_unicode(p)
    return gn in pn or pn in gn


def match_invoice_number(gold, pred) -> bool:
    g = _clean(gold).lstrip("0")
    p = _clean(pred).lstrip("0")
    return bool(g) and g == p


def match_date(gold, pred) -> bool:
    g, p = _clean(gold), _clean(pred)
    if not g or not p:
        return False
    if g == p:
        return True
    g_year = re.search(r'\b(\d{4})\b', g)
    p_year = re.search(r'\b(\d{4})\b', p)
    if g_year and p_year and g_year.group(1) != p_year.group(1):
        return False
    g_nums = re.findall(r'\d+', g)
    p_nums = re.findall(r'\d+', p)
    if len(g_nums) >= 3 and len(p_nums) >= 3:
        return sorted(g_nums[:3]) == sorted(p_nums[:3])
    return False


def match_amount(gold, pred) -> bool:
    g, p = _to_float(gold), _to_float(pred)
    if g is None or p is None:
        return False
    return abs(g - p) <= max(0.01, 0.01 * abs(g))


MATCH_FN = {
    "invoice_number":  match_invoice_number,
    "invoice_date":    match_date,
    "due_date":        match_date,
    "issuer_name":     match_string,
    "recipient_name":  match_string,
    "total_amount":    match_amount,
}


# ---------------------------------------------------------------------------
# Run pipeline on one file using a specific extractor class
# ---------------------------------------------------------------------------

def run_one(file_path: Path, InvoiceExtractor) -> dict:
    """
    Extract text and classify via run.py, then apply the given
    InvoiceExtractor for field extraction (bypasses run.py's fixed import).
    """
    from run import extract_text, classify_document, clean_text

    raw_text = extract_text(str(file_path))
    if not raw_text or not raw_text.strip():
        return {}

    cleaned = clean_text(raw_text)
    classification = classify_document(raw_text, cleaned)

    if classification["predicted_class"] != "invoice":
        return {"_classification": classification["predicted_class"]}

    extractor = InvoiceExtractor()
    # Pass file_path as optional kwarg — regex extractor ignores it,
    # image-based extractors (LayoutLMv3) use it to access the document image
    return extractor.extract(raw_text, file_path=str(file_path))


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(invoices_dir: Path, ground_truth: dict, InvoiceExtractor) -> dict:
    per_file = {}
    field_correct = {f: 0 for f in FIELDS}
    field_total   = {f: 0 for f in FIELDS}

    invoice_files = sorted(
        p for p in invoices_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    )

    if not invoice_files:
        print(f"No invoice files found in {invoices_dir}")
        sys.exit(1)

    for file_path in invoice_files:
        name = file_path.name

        # Exact match first, then try stripping trailing _N from stem
        # e.g. "test_invoice_1.pdf" → try "test_invoice.pdf"
        gt = ground_truth.get(name)
        if gt is None:
            alt = re.sub(r'_\d+(\.[^.]+)$', r'\1', name)
            gt = ground_truth.get(alt)

        if gt is None:
            print(f"  [SKIP] {name} — no ground truth entry")
            continue

        print(f"  Running: {name} ...", end=" ", flush=True)
        try:
            extracted = run_one(file_path, InvoiceExtractor)
        except Exception as e:
            print(f"ERROR: {e}")
            per_file[name] = {"error": str(e)}
            continue

        if "_classification" in extracted:
            cls = extracted["_classification"]
            print(f"classified as '{cls}' (not invoice) — 0/6")
            per_file[name] = {"error": f"classified as {cls}"}
            continue

        file_result = {}
        correct_count = 0
        n_gold = 0

        for field in FIELDS:
            gold_val = gt.get(field)
            pred_val = extracted.get(field)

            if gold_val is not None:
                is_correct = MATCH_FN[field](gold_val, pred_val)
                field_correct[field] += int(is_correct)
                field_total[field]   += 1
                correct_count += int(is_correct)
                n_gold += 1
            else:
                is_correct = None

            file_result[field] = {
                "gold":      gold_val,
                "predicted": pred_val,
                "correct":   is_correct,
            }

        print(f"{correct_count}/{n_gold} correct")
        per_file[name] = file_result

    field_summary = {}
    for field in FIELDS:
        total   = field_total[field]
        correct = field_correct[field]
        field_summary[field] = {
            "correct":  correct,
            "total":    total,
            "accuracy": round(correct / total, 3) if total else None,
        }

    return {"per_file": per_file, "field_summary": field_summary}


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(results: dict, extractor_label: str) -> None:
    summary = results["field_summary"]

    print()
    print(f"  Extractor: {extractor_label}")
    print("=" * 55)
    print("  Per-field accuracy")
    print("=" * 55)
    print(f"  {'Field':<20} {'Correct':>7}  {'Total':>5}  {'Accuracy':>9}")
    print("  " + "-" * 48)

    total_correct = 0
    total_fields  = 0
    for field, stats in summary.items():
        c   = stats["correct"]
        t   = stats["total"]
        acc = stats["accuracy"]
        bar = "█" * int((acc or 0) * 10)
        acc_str = f"{acc:.1%}" if acc is not None else "  N/A"
        print(f"  {field:<20} {c:>7}  {t:>5}  {acc_str:>9}  {bar}")
        total_correct += c
        total_fields  += t

    overall = total_correct / total_fields if total_fields else 0
    print("  " + "-" * 48)
    print(f"  {'OVERALL':<20} {total_correct:>7}  {total_fields:>5}  {overall:.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate invoice extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_pipeline.py
  python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v1.py
  python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v0.py --output results/eval_v0.json
        """
    )
    parser.add_argument(
        "--extractor",
        default="scripts/extract_invoice_fields.py",
        help="Path to extractor .py file to test (default: scripts/extract_invoice_fields.py)"
    )
    parser.add_argument(
        "--invoices", default="tests/invoices",
        help="Directory of test invoice files (default: tests/invoices/)"
    )
    parser.add_argument(
        "--ground-truth", default="tests/ground_truth.json",
        help="JSON file with ground truth (default: tests/ground_truth.json)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results JSON to this path"
    )
    args = parser.parse_args()

    extractor_path = PROJ_ROOT / args.extractor
    invoices_dir   = PROJ_ROOT / args.invoices
    gt_path        = PROJ_ROOT / args.ground_truth

    for path, label in [
        (extractor_path, "extractor"),
        (invoices_dir,   "invoices directory"),
        (gt_path,        "ground truth"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print(f"Loading extractor: {extractor_path.name}")
    InvoiceExtractor = load_extractor(extractor_path)

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    n_files = sum(
        1 for p in invoices_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    )
    print(f"Evaluating {n_files} files in {invoices_dir.name}/")
    print()

    results = evaluate(invoices_dir, ground_truth, InvoiceExtractor)
    print_report(results, extractor_path.name)

    if args.output:
        out_path = PROJ_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()