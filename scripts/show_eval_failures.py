"""
Show Evaluation Failures — Detailed Breakdown

Reads a results JSON from evaluate_pipeline.py and prints exactly what
went wrong for each invoice, field by field.

Usage:
    python scripts/show_eval_failures.py results/eval_v1.json
    python scripts/show_eval_failures.py results/eval_v0.json results/eval_v1.json
"""

import json
import sys
from pathlib import Path

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]

FIELD_SHORT = {
    "invoice_number":  "inv_no",
    "invoice_date":    "inv_date",
    "due_date":        "due_date",
    "issuer_name":     "issuer",
    "recipient_name":  "recipient",
    "total_amount":    "total",
}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(results: dict, label: str) -> None:
    s = results["field_summary"]
    total_c = sum(v["correct"] for v in s.values())
    total_t = sum(v["total"]   for v in s.values())
    overall = total_c / total_t if total_t else 0
    print(f"  {label}: {total_c}/{total_t} ({overall:.1%})")
    for field, stats in s.items():
        acc = stats["accuracy"]
        bar = "█" * int((acc or 0) * 10)
        acc_str = f"{acc:.0%}" if acc is not None else " N/A"
        print(f"    {FIELD_SHORT[field]:<10} {stats['correct']:>3}/{stats['total']:<3} "
              f"{acc_str:>5}  {bar}")


def print_failures(results: dict, label: str) -> None:
    per_file = results.get("per_file", {})
    if not per_file:
        print("  No per-file data in this results file.")
        return

    print(f"\n{'='*70}")
    print(f"  {label} — per-invoice failures")
    print(f"{'='*70}")

    for fname, fields in sorted(per_file.items()):
        if "error" in fields:
            print(f"\n  {fname}: ERROR — {fields['error']}")
            continue

        failures = {f: v for f, v in fields.items()
                    if isinstance(v, dict) and v.get("correct") is False}
        correct  = sum(1 for v in fields.values()
                       if isinstance(v, dict) and v.get("correct") is True)
        total    = sum(1 for v in fields.values()
                       if isinstance(v, dict) and v.get("correct") is not None)

        if not failures:
            print(f"\n  ✓ {fname} — {correct}/{total} all correct")
            continue

        print(f"\n  ✗ {fname} — {correct}/{total} correct")
        for field, data in failures.items():
            gold = data.get("gold")
            pred = data.get("predicted")
            pred_str = f'"{pred}"' if pred else "None"
            print(f"    {FIELD_SHORT.get(field, field):<10}  "
                  f"expected: {str(gold)[:35]:<37}  "
                  f"got: {pred_str[:35]}")


def compare_two(r0: dict, r1: dict, label0: str, label1: str) -> None:
    """Show fields that improved or regressed between two versions."""
    s0 = r0["field_summary"]
    s1 = r1["field_summary"]

    print(f"\n{'='*70}")
    print(f"  Comparison: {label0}  →  {label1}")
    print(f"{'='*70}")
    print(f"  {'Field':<15} {label0:>8}  {label1:>8}  {'Change':>8}")
    print(f"  " + "-"*50)

    total_delta = 0
    for field in FIELDS:
        a0 = s0.get(field, {}).get("accuracy") or 0
        a1 = s1.get(field, {}).get("accuracy") or 0
        delta = a1 - a0
        total_delta += delta
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        col   = "" if delta == 0 else ("+" if delta > 0 else "")
        print(f"  {FIELD_SHORT[field]:<15} {a0:>7.1%}  {a1:>7.1%}  "
              f"{col}{delta:>+6.1%} {arrow}")

    # Overall
    t0c = sum(v["correct"] for v in s0.values())
    t0t = sum(v["total"]   for v in s0.values())
    t1c = sum(v["correct"] for v in s1.values())
    t1t = sum(v["total"]   for v in s1.values())
    ov0 = t0c/t0t if t0t else 0
    ov1 = t1c/t1t if t1t else 0
    print(f"  " + "-"*50)
    print(f"  {'OVERALL':<15} {ov0:>7.1%}  {ov1:>7.1%}  {ov1-ov0:>+6.1%} "
          f"{'▲' if ov1>ov0 else '▼'}")

    # Files that got worse
    per0 = r0.get("per_file", {})
    per1 = r1.get("per_file", {})
    regressions = []
    for fname in per0:
        if fname not in per1:
            continue
        c0 = sum(1 for v in per0[fname].values()
                 if isinstance(v, dict) and v.get("correct") is True)
        c1 = sum(1 for v in per1[fname].values()
                 if isinstance(v, dict) and v.get("correct") is True)
        if c1 < c0:
            regressions.append((fname, c0, c1))

    if regressions:
        print(f"\n  ⚠ Regressions ({len(regressions)} files got worse):")
        for fname, c0, c1 in regressions:
            print(f"    {fname}: {c0} → {c1} correct")
    else:
        print(f"\n  ✓ No regressions — v1 never made a previously correct field wrong")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_eval_failures.py <results.json> [results2.json]")
        sys.exit(1)

    path0 = sys.argv[1]
    r0 = load(path0)
    label0 = Path(path0).stem

    print(f"\nSummary")
    print_summary(r0, label0)

    if len(sys.argv) >= 3:
        path1 = sys.argv[2]
        r1 = load(path1)
        label1 = Path(path1).stem
        print_summary(r1, label1)
        compare_two(r0, r1, label0, label1)
        print_failures(r1, label1)
    else:
        print_failures(r0, label0)


if __name__ == "__main__":
    main()