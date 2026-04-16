#!/usr/bin/env python3
"""
Build line-level training tables from FATURA Original_Format annotations.

Approach:
- OCR each FATURA image with Tesseract
- group words into OCR lines
- use FATURA field boxes/text to assign line labels

Outputs:
- results/line/fatura_lines_train.csv
- results/line/fatura_lines_dev.csv
- results/line/fatura_lines_test.csv
- results/line/fatura_label_counts.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytesseract
from PIL import Image


PROJ_ROOT = Path(__file__).resolve().parents[2]
FATURA_DIR = PROJ_ROOT / "data" / "fatura"
IMG_DIR = FATURA_DIR / "images"
ANN_DIR = FATURA_DIR / "Annotations" / "Original_Format"
OUT_DIR = PROJ_ROOT / "results" / "line"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": FATURA_DIR / "strat1_train.csv",
    "dev": FATURA_DIR / "strat1_dev.csv",
    "test": FATURA_DIR / "strat1_test.csv",
}

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9:/.,#\- ]", "", s)
    return s.strip()


def text_match(a: str, b: str) -> bool:
    a = norm_text(a)
    b = norm_text(b)
    if not a or not b:
        return False
    return a in b or b in a


def parse_pdf_box_to_image_box(bbox_value, page_h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    FATURA bbox example:
    [[332.0, 614.4058], [500.432, 698.4058]]

    This appears to be PDF-style coordinates (origin at bottom-left),
    while OCR uses image coords (origin at top-left).
    """
    if not bbox_value or not isinstance(bbox_value, list) or len(bbox_value) != 2:
        return None

    try:
        x1, y1 = bbox_value[0]
        x2, y2 = bbox_value[1]
        left = float(min(x1, x2))
        right = float(max(x1, x2))
        top = float(page_h - max(y1, y2))
        bottom = float(page_h - min(y1, y2))
        return (left, top, right, bottom)
    except Exception:
        return None


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_vertical_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    _, ay1, _, ay2 = a
    _, by1, _, by2 = b
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(ay2, by2) - min(ay1, by1)
    if union <= 0:
        return 0.0
    return inter / union


def extract_recipient_name(buyer_text: str) -> Optional[str]:
    """
    Example:
    Bill to:Denise Perez
    16424 Timothy Mission
    ...
    -> Denise Perez
    """
    if not buyer_text:
        return None

    lines = [ln.strip() for ln in str(buyer_text).splitlines() if ln.strip()]
    if not lines:
        return None

    first = lines[0]
    first = re.sub(r"^(bill\s*to|buyer|customer|client)\s*:?\s*", "", first, flags=re.I).strip()
    if first:
        return first

    if len(lines) > 1:
        return lines[1].strip()

    return None


def extract_invoice_number_target(ann: Dict) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
    """
    FATURA may or may not include invoice number explicitly.
    We try INVOICE_INFO first.
    """
    info = ann.get("INVOICE_INFO")

    if isinstance(info, dict):
        text = info.get("text", "")
        bbox = info.get("bbox")
        if text and re.search(r"\d", text):
            return text, bbox

    if isinstance(info, list):
        for item in info:
            if isinstance(item, dict):
                text = item.get("text", "")
                bbox = item.get("bbox")
                if text and re.search(r"\d", text):
                    return text, bbox

    return None, None


def build_targets(ann: Dict, page_h: int) -> Dict[str, Dict]:
    targets: Dict[str, Dict] = {}

    # invoice date
    date_obj = ann.get("DATE")
    if isinstance(date_obj, dict) and date_obj.get("text"):
        targets["invoice_date"] = {
            "text": date_obj.get("text", ""),
            "bbox": parse_pdf_box_to_image_box(date_obj.get("bbox"), page_h),
        }

    # due date
    due_obj = ann.get("DUE_DATE")
    if isinstance(due_obj, dict) and due_obj.get("text"):
        targets["due_date"] = {
            "text": due_obj.get("text", ""),
            "bbox": parse_pdf_box_to_image_box(due_obj.get("bbox"), page_h),
        }

    # total
    total_obj = ann.get("TOTAL")
    if isinstance(total_obj, dict) and total_obj.get("text"):
        targets["total_amount"] = {
            "text": total_obj.get("text", ""),
            "bbox": parse_pdf_box_to_image_box(total_obj.get("bbox"), page_h),
        }

    # recipient / buyer
    buyer_obj = ann.get("BUYER")
    if isinstance(buyer_obj, dict) and buyer_obj.get("text"):
        recip_text = extract_recipient_name(buyer_obj.get("text", ""))
        if recip_text:
            targets["recipient_name"] = {
                "text": recip_text,
                "bbox": parse_pdf_box_to_image_box(buyer_obj.get("bbox"), page_h),
            }

    # invoice number (if present)
    inv_text, inv_bbox_raw = extract_invoice_number_target(ann)
    if inv_text:
        targets["invoice_number"] = {
            "text": inv_text,
            "bbox": parse_pdf_box_to_image_box(inv_bbox_raw, page_h) if inv_bbox_raw else None,
        }

    # issuer_name is NOT clearly annotated in FATURA Original_Format
    # leave it absent for now
    return targets


def group_ocr_to_lines(data: Dict[str, List], page_w: int, page_h: int) -> pd.DataFrame:
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

        rows.append(
            {
                "text": text,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "right": int(data["left"][i]) + int(data["width"][i]),
                "bottom": int(data["top"][i]) + int(data["height"][i]),
                "conf": conf,
                "block_num": int(data["block_num"][i]),
                "par_num": int(data["par_num"][i]),
                "line_num": int(data["line_num"][i]),
                "word_num": int(data["word_num"][i]),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grouped = []

    for (_, _, _), grp in df.groupby(["block_num", "par_num", "line_num"], sort=False):
        grp = grp.sort_values("left")
        line_text = " ".join(grp["text"].tolist()).strip()

        grouped.append(
            {
                "line_text": line_text,
                "left": int(grp["left"].min()),
                "top": int(grp["top"].min()),
                "right": int(grp["right"].max()),
                "bottom": int(grp["bottom"].max()),
                "width": int(grp["right"].max() - grp["left"].min()),
                "height": int(grp["bottom"].max() - grp["top"].min()),
                "token_count": int(len(grp)),
                "page_width": int(page_w),
                "page_height": int(page_h),
            }
        )

    out = pd.DataFrame(grouped).sort_values(["top", "left"]).reset_index(drop=True)
    out["left_rel"] = out["left"] / out["page_width"].clip(lower=1)
    out["top_rel"] = out["top"] / out["page_height"].clip(lower=1)
    out["right_rel"] = out["right"] / out["page_width"].clip(lower=1)
    out["bottom_rel"] = out["bottom"] / out["page_height"].clip(lower=1)
    return out


def line_positive_for_field(line_row: pd.Series, field: str, target: Optional[Dict]) -> int:
    if not target:
        return 0

    line_text = str(line_row["line_text"])
    target_text = str(target.get("text", "")).strip()
    target_bbox = target.get("bbox")

    line_bbox = (
        float(line_row["left"]),
        float(line_row["top"]),
        float(line_row["right"]),
        float(line_row["bottom"]),
    )

    # Primary match: text
    if field == "recipient_name":
        if target_text and text_match(target_text, line_text):
            return 1
        return 0

    # For other fields, allow text or geometric overlap
    if target_text and text_match(target_text, line_text):
        return 1

    if target_bbox is not None:
        iou = bbox_iou(line_bbox, target_bbox)
        vov = bbox_vertical_overlap(line_bbox, target_bbox)
        if iou >= 0.15 or vov >= 0.60:
            return 1

    return 0


def process_one_document(split_name: str, idx: int, img_name: str, annot_name: str) -> List[Dict]:
    img_path = IMG_DIR / img_name
    ann_path = ANN_DIR / annot_name

    if not img_path.exists():
        print(f"[WARN] Missing image: {img_path}")
        return []
    if not ann_path.exists():
        print(f"[WARN] Missing annotation: {ann_path}")
        return []

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    img = Image.open(img_path).convert("RGB")
    page_w, page_h = img.size

    targets = build_targets(ann, page_h)

    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    line_df = group_ocr_to_lines(ocr, page_w, page_h)
    if line_df.empty:
        return []

    doc_id = f"fatura_{split_name}_{idx:05d}"
    rows = []

    for line_idx, row in line_df.iterrows():
        out = {
            "split": split_name,
            "doc_id": doc_id,
            "file_name": img_name,
            "line_id": f"{doc_id}_line_{line_idx:03d}",
            "line_idx": int(line_idx),
            "line_text": row["line_text"],
            "left": int(row["left"]),
            "top": int(row["top"]),
            "right": int(row["right"]),
            "bottom": int(row["bottom"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "page_width": int(row["page_width"]),
            "page_height": int(row["page_height"]),
            "left_rel": float(row["left_rel"]),
            "top_rel": float(row["top_rel"]),
            "right_rel": float(row["right_rel"]),
            "bottom_rel": float(row["bottom_rel"]),
            "token_count": int(row["token_count"]),
        }

        for field in FIELDS:
            out[f"y_{field}"] = line_positive_for_field(row, field, targets.get(field))

        rows.append(out)

    return rows


def build_split(split_name: str, split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    rows: List[Dict] = []

    print(f"[INFO] Building split: {split_name} ({len(df)} docs)")

    for idx, record in df.iterrows():
        img_name = str(record["img_path"]).strip()
        annot_name = str(record["annot_path"]).strip()
        rows.extend(process_one_document(split_name, idx, img_name, annot_name))

        if (idx + 1) % 250 == 0:
            print(f"  processed {idx + 1}/{len(df)}")

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"fatura_lines_{split_name}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] {split_name}: {len(out_df)} line rows -> {out_path}")
    return out_df


def main():
    summary: Dict[str, Dict[str, int]] = {}

    for split_name, split_csv in SPLIT_FILES.items():
        if not split_csv.exists():
            print(f"[WARN] Missing split file: {split_csv}")
            continue

        out_df = build_split(split_name, split_csv)

        split_stats = {
            "docs": int(out_df["doc_id"].nunique()) if not out_df.empty else 0,
            "rows": int(len(out_df)),
        }
        for field in FIELDS:
            col = f"y_{field}"
            split_stats[field] = int(out_df[col].sum()) if col in out_df.columns else 0

        summary[split_name] = split_stats

    summary_path = OUT_DIR / "fatura_label_counts.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Summary -> {summary_path}")


if __name__ == "__main__":
    main()