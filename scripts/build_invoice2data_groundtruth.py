"""
Convert invoice2data test suite (tests/compare/) to our ground_truth.json format.

invoice2data JSONs contain: invoice_number, date, amount, (sometimes issuer/desc)
We map these to our 6-field schema, leaving due_date and recipient_name as null.

Usage (run from silly-stats root):
    python scripts/build_invoice2data_groundtruth.py \
        --compare-dir ../invoice2data/tests/compare \
        --output tests/ground_truth_invoice2data.json

Then evaluate:
    python scripts/evaluate_pipeline.py \
        --invoices ../invoice2data/tests/compare \
        --ground-truth tests/ground_truth_invoice2data.json \
        --extractor scripts/extract_invoice_fields_v3.py \
        --output results/eval_v3_invoice2data.json
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


def parse_date(val) -> str | None:
    """Normalise invoice2data date to YYYY-MM-DD."""
    if not val:
        return None
    s = str(val)
    # Already ISO
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # datetime repr: datetime.date(2014, 5, 7)
    m = re.match(r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)', s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # Try parsing common formats
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%B %d, %Y'):
        try:
            return datetime.strptime(s.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return s.strip()


def parse_amount(val) -> str | None:
    if val is None:
        return None
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        cleaned = re.sub(r'[^\d\.]', '', str(val))
        return f"{float(cleaned):.2f}" if cleaned else None


def convert(compare_dir: Path, output_path: Path) -> None:
    ground_truth = {}
    skipped = []

    json_files = sorted(compare_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {compare_dir}")

    for json_path in json_files:
        # Find corresponding PDF or PNG or TXT
        stem = json_path.stem
        pdf_path = None
        for ext in ['.pdf', '.png', '.txt']:
            candidate = compare_dir / (stem + ext)
            if candidate.exists():
                pdf_path = candidate
                break

        if pdf_path is None:
            print(f"  [SKIP] {json_path.name} — no matching PDF/PNG/TXT found")
            skipped.append(json_path.name)
            continue

        with open(json_path, encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  [SKIP] {json_path.name} — invalid JSON")
                skipped.append(json_path.name)
                continue

        # invoice2data JSON can contain a list (multiple results) or a dict
        if isinstance(data, list):
            if not data:
                skipped.append(json_path.name)
                continue
            data = data[0]

        # Extract fields — invoice2data uses various key names
        inv_no   = data.get('invoice_number') or data.get('invoice_no') or data.get('number')
        inv_date = parse_date(data.get('date') or data.get('invoice_date'))
        total    = parse_amount(data.get('amount') or data.get('total') or data.get('total_amount'))
        issuer   = data.get('issuer') or data.get('vendor') or data.get('supplier')

        # At least one field must be present to be useful
        if not any([inv_no, inv_date, total]):
            print(f"  [SKIP] {stem} — no usable fields in JSON")
            skipped.append(json_path.name)
            continue

        entry = {
            "invoice_number": str(inv_no).strip() if inv_no else None,
            "invoice_date":   inv_date,
            "due_date":       None,    # not in invoice2data
            "issuer_name":    str(issuer).strip() if issuer else None,
            "recipient_name": None,    # not in invoice2data
            "total_amount":   total,
        }

        ground_truth[pdf_path.name] = entry
        fields_found = sum(1 for v in entry.values() if v is not None)
        print(f"  ✓ {pdf_path.name:<35} {fields_found}/6 fields: "
              f"inv_no={'✓' if inv_no else '✗'} "
              f"date={'✓' if inv_date else '✗'} "
              f"total={'✓' if total else '✗'} "
              f"issuer={'✓' if issuer else '✗'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print()
    print(f"Converted {len(ground_truth)} invoices → {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")
    print()
    print("To evaluate:")
    print(f"  python scripts/evaluate_pipeline.py \\")
    print(f"    --invoices {compare_dir} \\")
    print(f"    --ground-truth {output_path} \\")
    print(f"    --extractor scripts/extract_invoice_fields_v3.py \\")
    print(f"    --output results/eval_v3_invoice2data.json")


def main():
    parser = argparse.ArgumentParser(
        description="Convert invoice2data test JSONs to silly-stats ground truth format"
    )
    parser.add_argument(
        "--compare-dir",
        default="../invoice2data/tests/compare",
        help="Path to invoice2data/tests/compare/ directory"
    )
    parser.add_argument(
        "--output",
        default="tests/ground_truth_invoice2data.json",
        help="Output ground truth JSON path"
    )
    args = parser.parse_args()

    proj_root   = Path(__file__).resolve().parent.parent
    compare_dir = (proj_root / args.compare_dir).resolve()
    output_path = proj_root / args.output

    if not compare_dir.exists():
        print(f"ERROR: compare dir not found: {compare_dir}")
        print("Make sure you cloned invoice2data next to silly-stats:")
        print("  Statistical_learning/")
        print("  ├── silly-stats/")
        print("  └── invoice2data/")
        raise SystemExit(1)

    convert(compare_dir, output_path)


if __name__ == "__main__":
    main()
