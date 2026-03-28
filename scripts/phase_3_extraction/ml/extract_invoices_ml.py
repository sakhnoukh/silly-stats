"""
Phase 3 — ML Extraction (Method C)

Applies trained field-ranker models to the pre-built candidate tables to
produce top-1 field extractions per document.

For each field, loads the candidate CSV, scores every candidate using the
trained ranker, and picks the highest-scoring one as the extracted value.

Inputs:
  - results/phase3_extraction/candidates/candidates_<field>.csv
  - models/phase3_extraction/<field>_ranker.pkl

Outputs:
  - results/phase3_extraction/ml/invoice_extractions_ml.csv
  - results/phase3_extraction/ml/invoice_extractions_ml.json
  - results/phase3_extraction/ml/extraction_coverage_ml.txt

Usage:
    python -m scripts.phase_3_extraction.ml.extract_invoices_ml
"""

from pathlib import Path
import json
import pickle

import pandas as pd

PROJ_ROOT    = Path(__file__).resolve().parents[3]
CANDS_DIR    = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"
MODELS_DIR   = PROJ_ROOT / "models" / "phase3_extraction"
RESULTS_DIR  = PROJ_ROOT / "results" / "phase3_extraction" / "ml"

BASELINE_CSV = PROJ_ROOT / "results" / "phase3_extraction" / "baseline" / "invoice_extractions.csv"

ML_FIELDS = ["recipient_name", "invoice_number", "total_amount"]
ALL_FIELDS = ["invoice_number", "invoice_date", "due_date", "issuer_name", "recipient_name", "total_amount"]

FIELD_CONFIG = {
    "recipient_name": {
        "candidates_csv": CANDS_DIR  / "candidates_recipient_name.csv",
        "model_pkl":      MODELS_DIR / "recipient_name_ranker.pkl",
    },
    "invoice_number": {
        "candidates_csv": CANDS_DIR  / "candidates_invoice_number.csv",
        "model_pkl":      MODELS_DIR / "invoice_number_ranker.pkl",
    },
    "total_amount": {
        "candidates_csv": CANDS_DIR  / "candidates_total_amount.csv",
        "model_pkl":      MODELS_DIR / "total_amount_ranker.pkl",
    },
}

META_COLUMNS = {"doc_id", "file_name", "_split", "field", "candidate_text",
                "label", "notes", "candidate_source"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ranker(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def make_feature_dicts(df: pd.DataFrame, numeric_cols, categorical_cols):
    rows = []
    for _, row in df.iterrows():
        feat = {}
        for col in numeric_cols:
            val = row.get(col)
            feat[col] = 0.0 if pd.isna(val) else float(val)
        for col in categorical_cols:
            val = row.get(col)
            feat[col] = "__MISSING__" if pd.isna(val) else str(val)
        rows.append(feat)
    return rows


def predict_top1(ranker: dict, candidates_df: pd.DataFrame) -> pd.Series:
    """
    For each doc_id in candidates_df, score all candidates and return
    a Series mapping doc_id -> best candidate_text (or None if no candidates).
    """
    pipeline      = ranker["pipeline"]
    numeric_cols  = ranker["numeric_cols"]
    categorical_cols = ranker["categorical_cols"]

    # Keep only feature cols that exist in this candidate table
    numeric_cols  = [c for c in numeric_cols  if c in candidates_df.columns]
    categorical_cols = [c for c in categorical_cols if c in candidates_df.columns]

    results = {}

    for doc_id, group in candidates_df.groupby("doc_id"):
        group = group.reset_index(drop=True)
        feat_dicts = make_feature_dicts(group, numeric_cols, categorical_cols)

        try:
            probs = pipeline.predict_proba(feat_dicts)[:, 1]
        except Exception:
            # Fallback to decision function if predict_proba fails
            try:
                probs = pipeline.decision_function(feat_dicts)
            except Exception:
                probs = [0.0] * len(group)

        best_idx = int(pd.Series(probs).idxmax())
        best_prob = probs[best_idx]

        # Require the model to be at least somewhat confident
        if best_prob >= 0.3:
            results[doc_id] = group.iloc[best_idx]["candidate_text"]
        else:
            results[doc_id] = None

    return pd.Series(results, name="ml_extracted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 — ML Extraction (Method C)")
    print("=" * 60)

    # Start from baseline to inherit: doc_id, file_name, _split, invoice_date,
    # due_date, issuer_name, payment_terms, scores — which ML doesn't cover.
    baseline = pd.read_csv(BASELINE_CSV)
    result = baseline[["doc_id", "file_name", "_split"]].copy()

    # Inherit fields not covered by ML from baseline
    for col in ["invoice_date", "due_date", "issuer_name", "payment_terms",
                "document_confidence", "invoice_like_score", "ocr_quality_score"]:
        if col in baseline.columns:
            result[col] = baseline[col]

    # Initialize ML fields as null (will fill per-field below)
    for field in ML_FIELDS:
        result[field] = None

    # Per-field ML extraction
    for field in ML_FIELDS:
        cfg = FIELD_CONFIG[field]
        cands_path = cfg["candidates_csv"]
        model_path = cfg["model_pkl"]

        if not cands_path.exists():
            print(f"  [{field}] candidates CSV not found — skipping")
            continue
        if not model_path.exists():
            print(f"  [{field}] ranker model not found — skipping")
            continue

        print(f"\n[{field}]")
        ranker = load_ranker(model_path)
        candidates = pd.read_csv(cands_path)

        top1 = predict_top1(ranker, candidates)
        filled = top1.notna().sum()
        print(f"  {len(candidates)} candidates across {candidates['doc_id'].nunique()} docs")
        print(f"  Extracted values: {filled}")

        # Merge into result (doc_id as key)
        top1_df = top1.reset_index()
        top1_df.columns = ["doc_id", field]
        result = result.drop(columns=[field], errors="ignore")
        result = result.merge(top1_df, on="doc_id", how="left")

    # Final column order
    col_order = (
        ["doc_id", "file_name", "_split"]
        + ALL_FIELDS
        + ["payment_terms", "document_confidence", "invoice_like_score", "ocr_quality_score"]
    )
    result = result.reindex(columns=[c for c in col_order if c in result.columns])

    # Save CSV + JSON
    out_csv  = RESULTS_DIR / "invoice_extractions_ml.csv"
    out_json = RESULTS_DIR / "invoice_extractions_ml.json"
    out_txt  = RESULTS_DIR / "extraction_coverage_ml.txt"

    result.to_csv(out_csv, index=False)
    result.to_json(out_json, orient="records", indent=2, force_ascii=False)

    # Coverage report
    lines = ["Phase 3 — ML Extraction Coverage Report", "=" * 50, ""]
    lines.append(f"Documents in output: {len(result)}")
    lines.append("")
    for field in ALL_FIELDS:
        if field not in result.columns:
            continue
        n = result[field].notna().sum()
        pct = 100 * n / len(result) if len(result) else 0
        source = "(ML)" if field in ML_FIELDS else "(baseline)"
        lines.append(f"  {field:20s}: {n:3d}/{len(result)} ({pct:5.1f}%) {source}")

    cov_text = "\n".join(lines)
    out_txt.write_text(cov_text, encoding="utf-8")

    print(f"\n{cov_text}")
    print(f"\nSaved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_txt}")


if __name__ == "__main__":
    main()
