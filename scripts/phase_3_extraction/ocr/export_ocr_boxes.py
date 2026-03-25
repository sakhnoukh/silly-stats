"""
Phase 3 — Export OCR boxes for invoice documents

For each invoice image, run Tesseract image_to_data and save word-level OCR boxes.

Outputs:
  data/ocr_boxes/<file_stem>.csv

Usage:
  python scripts/export_ocr_boxes.py
"""

import os
import sys
import csv
import shutil
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image

from pathlib import Path
from scripts.phase_3_extraction.baselines import extract_invoices as base

PROJ_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BOX_DIR = DATA_DIR / "ocr_boxes"


def resolve_image_path(file_name: str):
    """
    Try common raw subfolders.
    """
    candidates = [
        RAW_DIR / "invoice" / file_name,
        RAW_DIR / file_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def export_boxes_for_image(image_path: Path, out_csv: Path):
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path

    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    rows = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        rows.append({
            "text": text,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
            "word_num": int(data["word_num"][i]),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text", "left", "top", "width", "height", "conf",
                "block_num", "par_num", "line_num", "word_num"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 60)
    print("Phase 3 — Export OCR Boxes")
    print("=" * 60)

    invoices = base.load_invoices()
    BOX_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    missing = 0
    failed = 0

    for _, row in invoices.iterrows():
        file_name = row["file_name"]
        image_path = resolve_image_path(file_name)

        if image_path is None:
            missing += 1
            continue

        out_csv = BOX_DIR / (Path(file_name).stem + ".csv")
        if out_csv.exists():
            done += 1
            continue

        try:
            export_boxes_for_image(image_path, out_csv)
            done += 1
        except Exception as e:
            failed += 1
            print(f"Failed on {file_name}: {e}")

    print(f"\nDone: {done}")
    print(f"Missing images: {missing}")
    print(f"Failed OCR box exports: {failed}")
    print(f"OCR boxes saved in: {BOX_DIR}")


if __name__ == "__main__":
    main()