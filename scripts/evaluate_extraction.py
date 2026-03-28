"""
Phase 3 — Fair Evaluation of All Extraction Methods

Compares Method A (regex baseline), B (template), C (ML ranker), and
D (layout) against the gold dataset for three fields:
  - recipient_name
  - invoice_number
  - total_amount

Matching is intentionally lenient:
  - Strings are compared case-insensitively after stripping whitespace.
  - For recipient_name: a match if gold is a substring of extracted or
    vice-versa (accounts for OCR noise and partial captures).
  - For total_amount: normalize to float, compare numerically.
  - For invoice_number: case-insensitive exact match after stripping
    leading zeros and whitespace.

Outputs:
  - results/method_comparison.csv        per-method per-field accuracy
  - results/method_comparison_detail.csv per-doc per-method correctness
  - results/method_comparison.txt        human-readable report

Usage:
    python scripts/evaluate_extraction.py
"""

from pathlib import Path
import re
import json

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]

GOLD_CSV  = PROJ_ROOT / "results" / "gold_dataset.csv"

EXTRACTION_FILES = {
    "baseline":  PROJ_ROOT / "results" / "phase3_extraction" / "baseline"  / "invoice_extractions.csv",
    "template":  PROJ_ROOT / "results" / "phase3_extraction" / "template"  / "invoice_extractions_template.csv",
    "layout":    PROJ_ROOT / "results" / "phase3_extraction" / "layout"    / "invoice_extractions_layout.csv",
    "ml":        PROJ_ROOT / "results" / "phase3_extraction" / "ml"        / "invoice_extractions_ml.csv",
}

EVAL_FIELDS = ["recipient_name", "invoice_number", "total_amount"]

OUT_DIR = PROJ_ROOT / "results"


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _clean_str(s) -> str:
    if pd.isna(s) or s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _to_float(s) -> float | None:
    if pd.isna(s) or s is None or str(s).strip() == "":
        return None
    cleaned = re.sub(r"[^\d\.]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return None


def match_recipient_name(gold, pred) -> bool:
    g = _clean_str(gold)
    p = _clean_str(pred)
    if not g or not p:
        return False
    # Accept if either contains the other (handles truncation / surrounding text)
    return g in p or p in g


def match_invoice_number(gold, pred) -> bool:
    g = _clean_str(gold).lstrip("0")
    p = _clean_str(pred).lstrip("0")
    return g == p and g != ""


def match_total_amount(gold, pred) -> bool:
    g = _to_float(gold)
    p = _to_float(pred)
    if g is None or p is None:
        return False
    # Allow 1% tolerance for floating-point / OCR rounding
    return abs(g - p) <= max(0.01, 0.01 * abs(g))


MATCH_FN = {
    "recipient_name": match_recipient_name,
    "invoice_number": match_invoice_number,
    "total_amount":   match_total_amount,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(gold: pd.DataFrame, predictions: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        doc_id, field, gold, predicted, correct
    for all docs that have a gold value.
    """
    preds_idx = predictions.set_index("doc_id")
    rows = []

    for field in EVAL_FIELDS:
        if field not in gold.columns:
            continue
        gold_field = gold[["doc_id", field]].dropna(subset=[field])

        for _, row in gold_field.iterrows():
            doc_id = row["doc_id"]
            gold_val = row[field]

            if doc_id in preds_idx.index and field in preds_idx.columns:
                pred_val = preds_idx.loc[doc_id, field]
            else:
                pred_val = None

            correct = MATCH_FN[field](gold_val, pred_val)
            rows.append({
                "method":    method_name,
                "doc_id":    doc_id,
                "field":     field,
                "gold":      gold_val,
                "predicted": pred_val,
                "correct":   int(correct),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(GOLD_CSV)
    print(f"Gold dataset: {len(gold)} documents")
    for field in EVAL_FIELDS:
        n = gold[field].notna().sum() if field in gold.columns else 0
        print(f"  {field:20s}: {n} docs with gold labels")
    print()

    all_detail = []
    method_summaries = []

    for method_name, csv_path in EXTRACTION_FILES.items():
        if not csv_path.exists():
            print(f"  [{method_name}] extraction file not found — skipping")
            continue

        preds = pd.read_csv(csv_path)
        detail = evaluate(gold, preds, method_name)
        all_detail.append(detail)

        # Per-field accuracy for this method
        for field in EVAL_FIELDS:
            sub = detail[detail["field"] == field]
            if len(sub) == 0:
                continue
            n_total    = len(sub)
            n_correct  = sub["correct"].sum()
            n_predicted = sub["predicted"].notna().sum()
            n_missed   = (sub["predicted"].isna() | (sub["predicted"] == "")).sum()

            method_summaries.append({
                "method":      method_name,
                "field":       field,
                "gold_docs":   n_total,
                "correct":     int(n_correct),
                "accuracy":    round(n_correct / n_total, 4) if n_total else 0,
                "predicted":   int(n_predicted),
                "missed":      int(n_missed),
            })

    # Combine all detail rows
    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    summary_df = pd.DataFrame(method_summaries)

    # Save
    detail_csv  = OUT_DIR / "method_comparison_detail.csv"
    summary_csv = OUT_DIR / "method_comparison.csv"
    report_txt  = OUT_DIR / "method_comparison.txt"

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # Human-readable report
    lines = [
        "Phase 3 — Extraction Method Comparison",
        "=" * 60,
        "",
        "  Accuracy = correct matches / total gold-labeled documents",
        "  (only docs with a gold label count toward the denominator)",
        "",
    ]

    for field in EVAL_FIELDS:
        lines.append(f"Field: {field}")
        lines.append("-" * 40)
        sub = summary_df[summary_df["field"] == field].sort_values("accuracy", ascending=False)
        for _, row in sub.iterrows():
            bar_n = int(row["accuracy"] * 20)
            bar = "#" * bar_n + "." * (20 - bar_n)
            lines.append(
                f"  {row['method']:10s}  [{bar}]  "
                f"{row['correct']:3d}/{row['gold_docs']:3d}  "
                f"acc={row['accuracy']:.1%}  "
                f"(missed={row['missed']})"
            )
        lines.append("")

    # Overall summary table
    if len(summary_df):
        pivot = summary_df.pivot_table(
            index="method", columns="field", values="accuracy", aggfunc="first"
        )
        pivot["avg_accuracy"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_accuracy", ascending=False)
        lines.append("Overall accuracy by method (averaged across fields with gold labels):")
        lines.append("-" * 60)
        lines.append(f"{'method':12s}  " + "  ".join(f"{f:20s}" for f in pivot.columns))
        for method, row in pivot.iterrows():
            vals = "  ".join(f"{v:.1%}" if not pd.isna(v) else "    N/A             " for v in row)
            lines.append(f"{method:12s}  {vals}")

    report = "\n".join(lines)
    report_txt.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {detail_csv}")
    print(f"  {report_txt}")


if __name__ == "__main__":
    main()
