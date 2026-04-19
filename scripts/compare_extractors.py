import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_ROOT / "results"
TMP_DIR = RESULTS_DIR / "_compare_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

EVALUATE_SCRIPT = PROJ_ROOT / "scripts" / "evaluate_pipeline.py"

EXTRACTORS = [
    "scripts/extract_invoice_fields_v0.py",
    "scripts/extract_invoice_fields_v1.py",
    "scripts/extract_invoice_fields_v2.py",
    "scripts/extract_invoice_fields_v3.py",
    "scripts/extract_invoice_fields_v4.py",
    "scripts/extract_invoice_fields_v5.py",
]

DATASETS = [
    {
        "name": "synthetic",
        "invoices": "tests/invoices",
        "ground_truth": "tests/ground_truth.json",
    },
    {
        "name": "ocr",
        "invoices": "tests/invoices_ocr",
        "ground_truth": "tests/ground_truth_ocr.json",
    },
    {
        "name": "real",
        "invoices": "tests/invoices_real",
        "ground_truth": "tests/ground_truth_real6.json",
    },
]


def run_eval(extractor: str, dataset: Dict[str, str]) -> Path:
    extractor_name = Path(extractor).stem
    out_path = TMP_DIR / f"{extractor_name}_{dataset['name']}.json"

    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--extractor",
        extractor,
        "--invoices",
        dataset["invoices"],
        "--ground-truth",
        dataset["ground_truth"],
        "--output",
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJ_ROOT)
    return out_path


def parse_eval_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    field_summary = data.get("field_summary", {})

    fields = [
        "invoice_number",
        "invoice_date",
        "due_date",
        "issuer_name",
        "recipient_name",
        "total_amount",
    ]

    row = {
        "result_file": path.name,
    }

    overall_correct = 0
    overall_total = 0

    for field in fields:
        fstats = field_summary.get(field, {})
        correct = fstats.get("correct")
        total = fstats.get("total")
        accuracy = fstats.get("accuracy")

        row[f"{field}_correct"] = correct
        row[f"{field}_total"] = total
        row[f"{field}_accuracy"] = accuracy

        if correct is not None:
            overall_correct += int(correct)
        if total is not None:
            overall_total += int(total)

    row["overall_correct"] = overall_correct
    row["overall_total"] = overall_total
    row["overall_accuracy"] = round(overall_correct / overall_total, 4) if overall_total else None

    return row


def main() -> None:
    summary_rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}

    for extractor in EXTRACTORS:
        extractor_name = Path(extractor).stem
        detailed[extractor_name] = {}

        for dataset in DATASETS:
            dataset_name = dataset["name"]
            out_path = run_eval(extractor, dataset)

            with open(out_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            detailed[extractor_name][dataset_name] = raw

            row = parse_eval_json(out_path)
            row["extractor"] = extractor_name
            row["dataset"] = dataset_name
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "extractor",
            "dataset",
            "overall_correct",
            "overall_total",
            "overall_accuracy",
            "invoice_number_accuracy",
            "invoice_date_accuracy",
            "due_date_accuracy",
            "issuer_name_accuracy",
            "recipient_name_accuracy",
            "total_amount_accuracy",
            "result_file",
        ]
    ]

    csv_path = RESULTS_DIR / "extractor_comparison_summary.csv"
    json_path = RESULTS_DIR / "extractor_comparison_detailed.json"

    summary_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {json_path}")


if __name__ == "__main__":
    main()