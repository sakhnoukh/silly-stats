"""
Before/After comparison: does the invoice override logic improve accuracy?

Runs two modes against tests/invoices_real/ + ground_truth_real6.json:
  - BEFORE: extract only if classifier predicted "invoice"
  - AFTER : extract if classifier predicted "invoice" OR override fires

Usage:
    python scripts/compare_override.py
"""

import json
import re
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from run import (
    extract_text,
    clean_text,
    classify_document,
    extract_invoice_fields,
    detect_invoice_signals,
    should_run_invoice_extraction,
)

INVOICES_DIR  = PROJ_ROOT / "tests" / "invoices_real"
GROUND_TRUTH  = PROJ_ROOT / "tests" / "ground_truth_real6.json"

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


# ---------------------------------------------------------------------------
# Matching helpers (same logic as evaluate_pipeline.py)
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


def match_string(gold, pred) -> bool:
    import unicodedata
    g, p = _clean(gold), _clean(pred)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    def strip_accents(s):
        return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii')
    gn, pn = strip_accents(g), strip_accents(p)
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
    "invoice_number": match_invoice_number,
    "invoice_date":   match_date,
    "due_date":       match_date,
    "issuer_name":    match_string,
    "recipient_name": match_string,
    "total_amount":   match_amount,
}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def score_file(file_path, gt, use_override: bool) -> dict:
    """Returns per-field results + classification info for one file."""
    raw_text = extract_text(str(file_path))
    if not raw_text or not raw_text.strip():
        return {"error": "no_text"}

    cleaned = clean_text(raw_text)
    classification = classify_document(raw_text, cleaned)
    predicted = classification["predicted_class"]

    # Decide whether to run extraction
    if use_override:
        decision = should_run_invoice_extraction(raw_text, classification)
        run_extraction = decision["run_extraction"]
        reason = decision["reason"]
    else:
        run_extraction = (predicted == "invoice")
        reason = "predicted_invoice" if run_extraction else "not_invoice"

    if not run_extraction:
        return {
            "classified_as": predicted,
            "reason": reason,
            "run_extraction": False,
            "fields": {f: {"gold": gt.get(f), "predicted": None, "correct": False}
                       for f in FIELDS if gt.get(f) is not None},
        }

    extracted = extract_invoice_fields(raw_text, file_path=str(file_path))

    fields = {}
    for field in FIELDS:
        gold_val = gt.get(field)
        if gold_val is None:
            continue
        pred_val = extracted.get(field)
        correct = MATCH_FN[field](gold_val, pred_val)
        fields[field] = {"gold": gold_val, "predicted": pred_val, "correct": correct}

    return {
        "classified_as": predicted,
        "reason": reason,
        "run_extraction": True,
        "fields": fields,
    }


def run_mode(ground_truth: dict, use_override: bool) -> dict:
    label = "AFTER  (with override)" if use_override else "BEFORE (no override) "
    print(f"\n{'='*62}")
    print(f"  Mode: {label}")
    print(f"{'='*62}")

    field_correct = {f: 0 for f in FIELDS}
    field_total   = {f: 0 for f in FIELDS}
    per_file = {}

    invoice_files = sorted(
        p for p in INVOICES_DIR.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    )

    for file_path in invoice_files:
        name = file_path.name
        gt = ground_truth.get(name)
        if gt is None:
            alt = re.sub(r'_\d+(\.[^.]+)$', r'\1', name)
            gt = ground_truth.get(alt)
        if gt is None:
            print(f"  [SKIP] {name} — no ground truth")
            continue

        print(f"  {name:<35}", end=" ", flush=True)
        try:
            result = score_file(file_path, gt, use_override)
        except Exception as e:
            print(f"ERROR: {e}")
            per_file[name] = {"error": str(e), "run_extraction": False, "fields": {}}
            continue

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            per_file[name] = result
            continue

        if not result["run_extraction"]:
            print(f"→ classified as '{result['classified_as']}' — 0/{len(result['fields'])} (skipped)")
        else:
            n_correct = sum(1 for v in result["fields"].values() if v["correct"])
            n_total   = len(result["fields"])
            reason_tag = " [OVERRIDE]" if result["reason"] == "invoice_override" else ""
            print(f"→ {n_correct}/{n_total}{reason_tag}")

        for field, info in result["fields"].items():
            field_total[field]   += 1
            field_correct[field] += int(info["correct"])

        per_file[name] = result

    return {"per_file": per_file, "field_correct": field_correct, "field_total": field_total}


def print_summary(before: dict, after: dict) -> None:
    print(f"\n{'='*62}")
    print("  SUMMARY — per-field accuracy")
    print(f"{'='*62}")
    print(f"  {'Field':<20} {'BEFORE':>8}  {'AFTER':>8}  {'Delta':>7}")
    print("  " + "-" * 52)

    total_before_c = total_after_c = total_n = 0

    for field in FIELDS:
        bc = before["field_correct"][field]
        bt = before["field_total"][field]
        ac = after["field_correct"][field]
        at = after["field_total"][field]

        b_pct = bc / bt if bt else 0
        a_pct = ac / at if at else 0
        delta = a_pct - b_pct
        sign  = "+" if delta > 0 else ""

        total_before_c += bc
        total_after_c  += ac
        total_n         = max(total_n, bt)  # same denominator across modes

        print(f"  {field:<20} {b_pct:>7.1%}   {a_pct:>7.1%}   {sign}{delta:>+.1%}")

    print("  " + "-" * 52)
    bt_all = sum(before["field_total"].values())
    at_all = sum(after["field_total"].values())
    b_overall = total_before_c / bt_all if bt_all else 0
    a_overall = total_after_c  / at_all if at_all else 0
    delta_overall = a_overall - b_overall
    sign = "+" if delta_overall > 0 else ""
    print(f"  {'OVERALL':<20} {b_overall:>7.1%}   {a_overall:>7.1%}   {sign}{delta_overall:>+.1%}")
    print()

    # Show which files changed behaviour (newly extracted via override)
    changed = []
    for name, after_result in after["per_file"].items():
        before_result = before["per_file"].get(name, {})
        if after_result.get("reason") == "invoice_override" and not before_result.get("run_extraction"):
            n_correct = sum(1 for v in after_result.get("fields", {}).values() if v["correct"])
            n_total   = len(after_result.get("fields", {}))
            changed.append((name, n_correct, n_total))

    if changed:
        print(f"  Documents recovered by override ({len(changed)}):")
        for name, nc, nt in changed:
            print(f"    {name:<35} {nc}/{nt} fields correct")
        print()
    else:
        print("  No documents recovered by override — classifier was already correct on all real invoices.")
        print()


def print_field_diffs(after: dict) -> None:
    """Show per-file field-level detail for incorrect predictions."""
    print(f"\n{'='*62}")
    print("  FIELD-LEVEL DIFFS (AFTER mode — wrong predictions)")
    print(f"{'='*62}")

    any_wrong = False
    for name, result in sorted(after["per_file"].items()):
        fields = result.get("fields", {})
        if not fields:
            continue
        wrongs = [(f, info) for f, info in fields.items() if not info["correct"]]
        if not wrongs:
            continue
        any_wrong = True
        n_correct = sum(1 for v in fields.values() if v["correct"])
        print(f"\n  {name} ({n_correct}/{len(fields)})")
        for f, info in wrongs:
            gold = info["gold"]
            pred = info["predicted"]
            print(f"    {f:<20} gold={gold!r:<30} pred={pred!r}")

    if not any_wrong:
        print("  All fields correct!")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not GROUND_TRUTH.exists():
        print(f"ERROR: ground truth not found at {GROUND_TRUTH}")
        sys.exit(1)

    if not INVOICES_DIR.exists():
        print(f"ERROR: invoices directory not found at {INVOICES_DIR}")
        sys.exit(1)

    with open(GROUND_TRUTH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    before_results = run_mode(ground_truth, use_override=False)
    after_results  = run_mode(ground_truth, use_override=True)

    print_summary(before_results, after_results)
    print_field_diffs(after_results)
