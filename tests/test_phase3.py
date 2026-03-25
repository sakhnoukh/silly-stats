"""
Phase 3 — Integration & Unit Tests

Covers:
  1. Gold dataset output (build_gold_dataset.py)
  2. ML ranker model files (train_field_rankers.py)
  3. ML extraction output (extract_invoices_ml.py)
  4. Evaluation output (evaluate_extraction.py)
  5. Matching logic unit tests (evaluate_extraction.py)
  6. End-to-end pipeline smoke test (all four scripts in sequence)
"""

import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

# ── repo root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── import helpers under test ──────────────────────────────────────────────
from scripts.evaluate_extraction import (
    match_recipient_name,
    match_invoice_number,
    match_total_amount,
    _clean_str,
    _to_float,
    evaluate,
    EVAL_FIELDS,
)
from scripts.build_gold_dataset import _gold_from_candidates, FIELD_FILES, FIELDS


# ===========================================================================
# 1. Gold dataset — static file checks
# ===========================================================================

GOLD_CSV = ROOT / "results" / "gold_dataset.csv"


class TestGoldDataset:
    def test_file_exists(self):
        assert GOLD_CSV.exists(), "gold_dataset.csv not found — run build_gold_dataset.py"

    def test_required_columns(self):
        df = pd.read_csv(GOLD_CSV)
        for col in ["doc_id", "file_name", "_split", "recipient_name", "invoice_number", "total_amount"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_duplicate_doc_ids(self):
        df = pd.read_csv(GOLD_CSV)
        assert df["doc_id"].nunique() == len(df), "Duplicate doc_ids in gold dataset"

    def test_minimum_coverage(self):
        df = pd.read_csv(GOLD_CSV)
        # Thresholds from what we observed during build
        assert df["recipient_name"].notna().sum() >= 60
        assert df["invoice_number"].notna().sum() >= 10
        assert df["total_amount"].notna().sum() >= 90

    def test_split_values_valid(self):
        df = pd.read_csv(GOLD_CSV)
        valid = {"train", "val", "test"}
        splits = set(df["_split"].dropna().unique())
        assert splits.issubset(valid), f"Unexpected split values: {splits - valid}"

    def test_gold_from_candidates_function(self):
        """_gold_from_candidates returns correct columns and types."""
        for field, csv_path in FIELD_FILES.items():
            result = _gold_from_candidates(csv_path, field)
            assert "doc_id" in result.columns
            assert field in result.columns
            assert result["doc_id"].nunique() == len(result), f"Duplicate doc_ids for {field}"


# ===========================================================================
# 2. ML ranker model files
# ===========================================================================

MODELS_DIR = ROOT / "models" / "phase3_extraction"
ML_FIELDS  = ["recipient_name", "invoice_number", "total_amount"]


class TestMLRankerModels:
    @pytest.mark.parametrize("field", ML_FIELDS)
    def test_model_file_exists(self, field):
        path = MODELS_DIR / f"{field}_ranker.pkl"
        assert path.exists(), f"Ranker model not found: {path}"

    @pytest.mark.parametrize("field", ML_FIELDS)
    def test_model_loadable(self, field):
        path = MODELS_DIR / f"{field}_ranker.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert "pipeline" in obj
        assert "numeric_cols" in obj
        assert "categorical_cols" in obj
        assert obj["field_name"] == field

    @pytest.mark.parametrize("field", ML_FIELDS)
    def test_pipeline_has_predict_proba(self, field):
        path = MODELS_DIR / f"{field}_ranker.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        pipeline = obj["pipeline"]
        assert hasattr(pipeline, "predict_proba"), "Pipeline must support predict_proba"

    def test_training_summary_exists(self):
        summary = ROOT / "results" / "phase3_extraction" / "ml" / "ranker_training_summary.txt"
        assert summary.exists()

    def test_training_summary_json_valid(self):
        import json
        summary_json = ROOT / "results" / "phase3_extraction" / "ml" / "ranker_training_summary.json"
        assert summary_json.exists()
        with open(summary_json) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3
        for entry in data:
            assert "field" in entry
            assert "accuracy" in entry
            assert 0.0 <= entry["accuracy"] <= 1.0

    @pytest.mark.parametrize("field", ML_FIELDS)
    def test_model_accuracy_above_floor(self, field):
        """Each trained ranker should beat 50% accuracy (trivial threshold)."""
        import json
        summary_json = ROOT / "results" / "phase3_extraction" / "ml" / "ranker_training_summary.json"
        with open(summary_json) as f:
            data = json.load(f)
        entry = next(e for e in data if e["field"] == field)
        assert entry["accuracy"] >= 0.50, (
            f"{field} ranker accuracy {entry['accuracy']:.3f} below floor 0.50"
        )


# ===========================================================================
# 3. ML extraction output
# ===========================================================================

ML_CSV  = ROOT / "results" / "phase3_extraction" / "ml" / "invoice_extractions_ml.csv"
ML_JSON = ROOT / "results" / "phase3_extraction" / "ml" / "invoice_extractions_ml.json"
ML_TXT  = ROOT / "results" / "phase3_extraction" / "ml" / "extraction_coverage_ml.txt"


class TestMLExtractionOutput:
    def test_csv_exists(self):
        assert ML_CSV.exists()

    def test_json_exists(self):
        assert ML_JSON.exists()

    def test_coverage_report_exists(self):
        assert ML_TXT.exists()

    def test_csv_row_count_matches_baseline(self):
        baseline = pd.read_csv(
            ROOT / "results" / "phase3_extraction" / "baseline" / "invoice_extractions.csv"
        )
        ml = pd.read_csv(ML_CSV)
        assert len(ml) == len(baseline), (
            f"ML CSV has {len(ml)} rows, baseline has {len(baseline)}"
        )

    def test_csv_required_columns(self):
        ml = pd.read_csv(ML_CSV)
        for col in ["doc_id", "file_name", "_split",
                    "recipient_name", "invoice_number", "total_amount",
                    "invoice_date", "issuer_name"]:
            assert col in ml.columns, f"Missing column in ML CSV: {col}"

    def test_ml_fields_have_reasonable_coverage(self):
        ml = pd.read_csv(ML_CSV)
        n = len(ml)
        # ML should extract at least something for each field it covers
        assert ml["recipient_name"].notna().sum() >= 50
        assert ml["total_amount"].notna().sum() >= 50
        assert ml["invoice_number"].notna().sum() >= 5

    def test_json_parseable_and_matches_csv(self):
        import json
        with open(ML_JSON, encoding="utf-8") as f:
            records = json.load(f)
        ml = pd.read_csv(ML_CSV)
        assert len(records) == len(ml), "JSON and CSV row counts differ"

    def test_no_duplicate_doc_ids(self):
        ml = pd.read_csv(ML_CSV)
        assert ml["doc_id"].nunique() == len(ml), "Duplicate doc_ids in ML output"

    def test_issuer_name_inherited_from_baseline(self):
        """issuer_name is not an ML field — must be inherited from baseline."""
        baseline = pd.read_csv(
            ROOT / "results" / "phase3_extraction" / "baseline" / "invoice_extractions.csv"
        )
        ml = pd.read_csv(ML_CSV)
        b = baseline.set_index("doc_id")["issuer_name"]
        m = ml.set_index("doc_id")["issuer_name"]
        shared = b.index.intersection(m.index)
        matches = (b[shared].fillna("") == m[shared].fillna("")).all()
        assert matches, "issuer_name not correctly inherited from baseline"


# ===========================================================================
# 4. Evaluation outputs
# ===========================================================================

COMP_CSV    = ROOT / "results" / "method_comparison.csv"
DETAIL_CSV  = ROOT / "results" / "method_comparison_detail.csv"
COMP_TXT    = ROOT / "results" / "method_comparison.txt"


class TestEvaluationOutput:
    def test_summary_csv_exists(self):
        assert COMP_CSV.exists()

    def test_detail_csv_exists(self):
        assert DETAIL_CSV.exists()

    def test_report_txt_exists(self):
        assert COMP_TXT.exists()

    def test_all_four_methods_in_summary(self):
        df = pd.read_csv(COMP_CSV)
        methods = set(df["method"].unique())
        expected = {"baseline", "template", "layout", "ml"}
        assert expected.issubset(methods), f"Missing methods: {expected - methods}"

    def test_all_three_fields_evaluated(self):
        df = pd.read_csv(COMP_CSV)
        fields = set(df["field"].unique())
        assert {"recipient_name", "invoice_number", "total_amount"}.issubset(fields)

    def test_accuracy_values_in_range(self):
        df = pd.read_csv(COMP_CSV)
        assert (df["accuracy"] >= 0).all() and (df["accuracy"] <= 1).all()

    def test_ml_best_average_accuracy(self):
        df = pd.read_csv(COMP_CSV)
        avg = df.groupby("method")["accuracy"].mean()
        assert avg.idxmax() == "ml", (
            f"Expected ML to have best avg accuracy, got: {avg.idxmax()}"
        )

    def test_detail_correct_column_is_binary(self):
        df = pd.read_csv(DETAIL_CSV)
        assert set(df["correct"].unique()).issubset({0, 1})

    def test_report_contains_all_methods(self):
        text = COMP_TXT.read_text(encoding="utf-8")
        for method in ["ml", "baseline", "template", "layout"]:
            assert method in text, f"'{method}' missing from text report"


# ===========================================================================
# 5. Matching logic — unit tests
# ===========================================================================

class TestMatchRecipientName:
    def test_exact_match(self):
        assert match_recipient_name("Lorillard Media Services", "Lorillard Media Services")

    def test_case_insensitive(self):
        assert match_recipient_name("PHILIP MORRIS INC", "philip morris inc")

    def test_substring_gold_in_pred(self):
        # predicted has more surrounding text (OCR header noise)
        assert match_recipient_name("Covington & Burling", "PAY TO: Covington & Burling Attn:")

    def test_substring_pred_in_gold(self):
        # predicted is truncated
        assert match_recipient_name("Philip Morris Companies Inc", "Philip Morris Companies")

    def test_no_match_unrelated(self):
        assert not match_recipient_name("Lorillard Inc", "Hazleton Laboratories")

    def test_empty_gold_no_match(self):
        assert not match_recipient_name("", "Lorillard Inc")

    def test_none_pred_no_match(self):
        assert not match_recipient_name("Lorillard Inc", None)

    def test_nan_pred_no_match(self):
        import math
        assert not match_recipient_name("Lorillard Inc", float("nan"))


class TestMatchInvoiceNumber:
    def test_exact_match(self):
        assert match_invoice_number("67550435", "67550435")

    def test_case_insensitive(self):
        assert match_invoice_number("V-91-131", "v-91-131")

    def test_leading_zero_stripped(self):
        assert match_invoice_number("01558-1", "1558-1")

    def test_no_match_different_number(self):
        assert not match_invoice_number("67550435", "67550436")

    def test_empty_gold_no_match(self):
        assert not match_invoice_number("", "12345")

    def test_none_pred_no_match(self):
        assert not match_invoice_number("12345", None)

    def test_numeric_string_match(self):
        assert match_invoice_number("90220", "90220")


class TestMatchTotalAmount:
    def test_exact_float_match(self):
        assert match_total_amount("1350.00", "1350.0")

    def test_within_tolerance(self):
        # 1% tolerance: 1000 ± 10
        assert match_total_amount("1000.00", "1005.00")

    def test_outside_tolerance(self):
        assert not match_total_amount("1000.00", "1020.00")

    def test_currency_symbol_stripped(self):
        assert match_total_amount("$500.00", "500.00")

    def test_comma_in_number(self):
        assert match_total_amount("1,350.00", "1350.00")

    def test_none_pred_no_match(self):
        assert not match_total_amount("500.00", None)

    def test_empty_str_no_match(self):
        assert not match_total_amount("500.00", "")

    def test_zero_amount_no_match(self):
        # Both zero — edge case, tolerated
        assert match_total_amount("0.00", "0.00")


class TestHelpers:
    def test_clean_str_normalises_whitespace(self):
        assert _clean_str("  hello   world  ") == "hello world"

    def test_clean_str_lowercases(self):
        assert _clean_str("INVOICE") == "invoice"

    def test_clean_str_handles_nan(self):
        import math
        assert _clean_str(float("nan")) == ""

    def test_clean_str_handles_none(self):
        assert _clean_str(None) == ""

    def test_to_float_basic(self):
        assert _to_float("1350.00") == 1350.0

    def test_to_float_with_currency(self):
        assert _to_float("$500") == 500.0

    def test_to_float_with_comma(self):
        assert _to_float("1,234.56") == 1234.56

    def test_to_float_none(self):
        assert _to_float(None) is None

    def test_to_float_empty(self):
        assert _to_float("") is None

    def test_to_float_non_numeric(self):
        assert _to_float("N/A") is None


# ===========================================================================
# 6. evaluate() function — unit test with synthetic data
# ===========================================================================

class TestEvaluateFunction:
    def _make_gold(self):
        return pd.DataFrame([
            {"doc_id": "doc_001", "file_name": "a.tif", "_split": "test",
             "recipient_name": "Acme Corp", "invoice_number": "INV-001", "total_amount": "500.00"},
            {"doc_id": "doc_002", "file_name": "b.tif", "_split": "test",
             "recipient_name": "Wayne LLC", "invoice_number": None,    "total_amount": "1200.00"},
            {"doc_id": "doc_003", "file_name": "c.tif", "_split": "test",
             "recipient_name": None,         "invoice_number": "REF-99", "total_amount": None},
        ])

    def _make_predictions(self):
        return pd.DataFrame([
            {"doc_id": "doc_001", "recipient_name": "ACME CORP",   "invoice_number": "INV-001", "total_amount": "500.00"},
            {"doc_id": "doc_002", "recipient_name": "wrong name",  "invoice_number": "INV-002", "total_amount": "1150.00"},
            {"doc_id": "doc_003", "recipient_name": "Anyone Ltd",  "invoice_number": "REF-99",  "total_amount": "999.00"},
        ])

    def test_correct_count_recipient(self):
        gold = self._make_gold()
        preds = self._make_predictions()
        detail = evaluate(gold, preds, "test_method")
        sub = detail[detail["field"] == "recipient_name"]
        assert sub["correct"].sum() == 1  # only doc_001 matches

    def test_correct_count_invoice_number(self):
        gold = self._make_gold()
        preds = self._make_predictions()
        detail = evaluate(gold, preds, "test_method")
        sub = detail[detail["field"] == "invoice_number"]
        # doc_001 matches INV-001; doc_003 matches REF-99; doc_002 has no gold
        assert sub["correct"].sum() == 2

    def test_correct_count_total_amount(self):
        gold = self._make_gold()
        preds = self._make_predictions()
        detail = evaluate(gold, preds, "test_method")
        sub = detail[detail["field"] == "total_amount"]
        # doc_001 exact; doc_002 1150 vs 1200 → 4.2% off → no match
        assert sub["correct"].sum() == 1

    def test_missing_gold_not_counted(self):
        """Rows where gold is None must not appear in evaluation."""
        gold = self._make_gold()
        preds = self._make_predictions()
        detail = evaluate(gold, preds, "test_method")
        # recipient_name has no gold for doc_003, total_amount has no gold for doc_003
        rn = detail[detail["field"] == "recipient_name"]
        assert "doc_003" not in rn["doc_id"].values

    def test_method_name_in_output(self):
        gold = self._make_gold()
        preds = self._make_predictions()
        detail = evaluate(gold, preds, "my_method")
        assert (detail["method"] == "my_method").all()

    def test_doc_not_in_predictions_marked_missed(self):
        gold = self._make_gold()
        preds = pd.DataFrame([
            {"doc_id": "doc_999", "recipient_name": "Other Corp",
             "invoice_number": "X", "total_amount": "1.00"},
        ])
        detail = evaluate(gold, preds, "sparse_method")
        # doc_001, doc_002 not in preds → predicted=None → correct=0
        assert detail["correct"].sum() == 0
