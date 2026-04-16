#!/usr/bin/env python3
"""
Line-level invoice field extractor.

Loads per-field line rankers trained on FATURA-derived line tables,
runs OCR on the input document, scores each OCR line, then parses the
best line for each field.

Works with:
python scripts/evaluate_pipeline.py --extractor scripts/line/extract_invoice_fields_linecls.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import pytesseract
from PIL import Image
from scipy.sparse import csr_matrix, hstack


PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

MODEL_DIR = PROJ_ROOT / "models" / "line"

FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "issuer_name",
    "recipient_name",
    "total_amount",
]


def _normalise_amount(raw: str) -> Optional[str]:
    if not raw:
        return None

    s = str(raw).strip()
    s = re.sub(r"[^\d,.\-]", "", s)

    if not s:
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if re.search(r",\d{2}$", s):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        val = float(s)
        if val <= 0:
            return None
        return f"{val:.2f}"
    except ValueError:
        return None


def _extract_first_date_candidate(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _normalise_date(raw: str) -> Optional[str]:
    s = _extract_first_date_candidate(raw)
    if not s:
        return None

    s = s.replace(".", "/").strip()

    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]

    cleaned = re.sub(r"\s+", " ", s.replace(",", ", ")).strip()
    cleaned = re.sub(r"\s+,", ",", cleaned)

    for fmt in fmts:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    if re.match(r"\d{4}-\d{2}-\d{2}$", s):
        return s
    return None


def _extract_invoice_number_from_line(line: str) -> Optional[str]:
    if not line:
        return None

    patterns = [
        r"(?:invoice|invoice\s*number|invoice\s*no|invoice\s*#|inv\s*#|inv\s*no)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_\-]{2,31})",
        r"\b([A-Z]{2,}[A-Z0-9/_\-]{2,31}\d[A-Z0-9/_\-]*)\b",
        r"\b([A-Z0-9]{3,}[/-][A-Z0-9/_\-]{2,31})\b",
        r"\b([A-Z0-9]{5,32})\b",
    ]

    for pat in patterns:
        m = re.search(pat, line, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if re.search(r"\d", val):
                return val
    return None


def _extract_amount_from_line(line: str) -> Optional[str]:
    if not line:
        return None

    candidates = re.findall(
        r"(?:[$€£¥]\s*)?\d[\d.,]{0,20}\d",
        line,
        flags=re.IGNORECASE,
    )
    candidates = [c.strip() for c in candidates if re.search(r"\d", c)]
    if not candidates:
        return None

    return _normalise_amount(candidates[-1])


def _clean_party_line(line: str) -> Optional[str]:
    if not line:
        return None

    s = str(line).strip()
    s = re.sub(r"^(bill\s*to|buyer|customer|client|ship\s*to|to|from|vendor|seller)\s*[:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" -:|")
    if len(s) < 3:
        return None
    return s


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    s = df["line_text"].fillna("")
    out = pd.DataFrame(index=df.index)
    out["has_digit"] = s.str.contains(r"\d", regex=True).astype(int)
    out["has_currency"] = s.str.contains(r"[$€£¥]|USD|EUR|GBP", regex=True, case=False).astype(int)
    out["has_date_word"] = s.str.contains(r"\b(date|due|invoice)\b", regex=True, case=False).astype(int)
    out["has_total_word"] = s.str.contains(r"\b(total|balance|amount|subtotal|vat|tax)\b", regex=True, case=False).astype(int)
    out["has_bill_word"] = s.str.contains(r"\b(bill|buyer|customer|client|ship)\b", regex=True, case=False).astype(int)
    out["uppercase_ratio"] = s.apply(
        lambda x: (sum(ch.isupper() for ch in x) / max(sum(ch.isalpha() for ch in x), 1))
    )
    out["digit_ratio"] = s.apply(
        lambda x: (sum(ch.isdigit() for ch in x) / max(len(x), 1))
    )
    return out


def build_numeric_features(df: pd.DataFrame) -> csr_matrix:
    base = pd.DataFrame(index=df.index)
    for col in [
        "left_rel",
        "top_rel",
        "right_rel",
        "bottom_rel",
        "width",
        "height",
        "token_count",
    ]:
        base[col] = df[col].fillna(0).astype(float)

    flags = add_text_flags(df)
    feats = pd.concat([base, flags], axis=1)
    return csr_matrix(feats.values.astype(float))


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
    groups = []

    for (_, _, _), grp in df.groupby(["block_num", "par_num", "line_num"], sort=False):
        grp = grp.sort_values("left")
        line_text = " ".join(grp["text"].tolist()).strip()
        groups.append(
            {
                "line_text": line_text,
                "left": int(grp["left"].min()),
                "top": int(grp["top"].min()),
                "right": int(grp["right"].max()),
                "bottom": int(grp["bottom"].max()),
                "width": int(grp["right"].max() - grp["left"].min()),
                "height": int(grp["bottom"].max() - grp["top"].min()),
                "token_count": int(len(grp)),
            }
        )

    out = pd.DataFrame(groups).sort_values(["top", "left"]).reset_index(drop=True)
    out["left_rel"] = out["left"] / max(page_w, 1)
    out["top_rel"] = out["top"] / max(page_h, 1)
    out["right_rel"] = out["right"] / max(page_w, 1)
    out["bottom_rel"] = out["bottom"] / max(page_h, 1)
    return out


class InvoiceExtractor:
    _cached_vectorizer = None
    _cached_models = None

    def __init__(self):
        if InvoiceExtractor._cached_vectorizer is None or InvoiceExtractor._cached_models is None:
            InvoiceExtractor._cached_vectorizer = joblib.load(MODEL_DIR / "line_vectorizer.pkl")
            models = {}
            for field in FIELDS:
                path = MODEL_DIR / f"{field}_line_ranker.pkl"
                if path.exists():
                    models[field] = joblib.load(path)
            InvoiceExtractor._cached_models = models

        self.vectorizer = InvoiceExtractor._cached_vectorizer
        self.models = InvoiceExtractor._cached_models

    def _file_to_image(self, file_path: str) -> Image.Image:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(file_path)
            page = doc[0]
            bm = page.render(scale=2.0)
            return bm.to_pil().convert("RGB")

        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return Image.open(file_path).convert("RGB")

        raise ValueError(f"Unsupported file type for line extractor: {ext}")

    def _make_line_df(self, invoice_text: str, file_path: Optional[str]) -> pd.DataFrame:
        if file_path is not None:
            ext = Path(file_path).suffix.lower()
            if ext in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                img = self._file_to_image(file_path)
                page_w, page_h = img.size
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                line_df = group_ocr_to_lines(data, page_w, page_h)
                if not line_df.empty:
                    return line_df

        raw_lines = [ln.strip() for ln in str(invoice_text).splitlines() if ln.strip()]
        if not raw_lines:
            return pd.DataFrame()

        rows = []
        n = max(len(raw_lines), 1)
        for i, line in enumerate(raw_lines):
            rows.append(
                {
                    "line_text": line,
                    "left": 0,
                    "top": i,
                    "right": len(line),
                    "bottom": i + 1,
                    "width": len(line),
                    "height": 1,
                    "token_count": len(line.split()),
                    "left_rel": 0.0,
                    "top_rel": i / n,
                    "right_rel": 1.0,
                    "bottom_rel": (i + 1) / n,
                }
            )
        return pd.DataFrame(rows)

    def _score_lines(self, line_df: pd.DataFrame):
        x_text = self.vectorizer.transform(line_df["line_text"].fillna("").astype(str))
        x_num = build_numeric_features(line_df)
        return hstack([x_text, x_num]).tocsr()

    def extract(self, invoice_text: str, file_path: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Safer hybrid extractor:
        - invoice_number: regex only
        - issuer_name: regex only
        - invoice_date: line model first, regex fallback
        - due_date: regex first, line fallback
        - recipient_name: regex first, line fallback
        - total_amount: regex first, line fallback
        """
        result = {field: None for field in FIELDS}

        # ------------------------------------------------------------
        # 1) Regex baseline
        # ------------------------------------------------------------
        try:
            from scripts.extract_invoice_fields_v3 import InvoiceExtractor as RegexExtractor
            regex_extractor = RegexExtractor()
            regex_result = regex_extractor.extract(invoice_text, file_path=file_path)
        except Exception:
            regex_result = {field: None for field in FIELDS}

        for field in FIELDS:
            result[field] = regex_result.get(field)

        # ------------------------------------------------------------
        # 2) Build line candidates
        # ------------------------------------------------------------
        line_df = self._make_line_df(invoice_text, file_path=file_path)
        if line_df.empty:
            return result

        x_text = self.vectorizer.transform(line_df["line_text"].fillna("").astype(str))
        x_num = build_numeric_features(line_df)
        x = hstack([x_text, x_num]).tocsr()

        def get_best_line(field: str) -> Optional[str]:
            model = self.models.get(field)
            if model is None:
                return None

            if hasattr(model, "decision_function"):
                scores = model.decision_function(x)
            else:
                scores = model.predict_proba(x)[:, 1]

            work = line_df.copy()
            work["score"] = scores
            work = work.sort_values("score", ascending=False).reset_index(drop=True)
            if work.empty:
                return None
            return str(work.loc[0, "line_text"]).strip()

        # ------------------------------------------------------------
        # 3) invoice_date: line first, regex fallback
        # ------------------------------------------------------------
        best_line = get_best_line("invoice_date")
        if best_line:
            line_val = _normalise_date(best_line)
            if line_val:
                result["invoice_date"] = line_val

        # ------------------------------------------------------------
        # 4) due_date: regex first, line fallback
        # ------------------------------------------------------------
        if not result.get("due_date"):
            best_line = get_best_line("due_date")
            if best_line:
                line_val = _normalise_date(best_line)
                if line_val and line_val != result.get("invoice_date"):
                    result["due_date"] = line_val

        # ------------------------------------------------------------
        # 5) recipient_name: regex first, line fallback
        # ------------------------------------------------------------
        if not result.get("recipient_name"):
            best_line = get_best_line("recipient_name")
            if best_line:
                line_val = _clean_party_line(best_line)
                if (
                    line_val
                    and not re.search(r"\b(invoice|date|due|total|amount|tax|vat|balance|subtotal)\b", line_val, flags=re.I)
                    and len(line_val) >= 3
                ):
                    result["recipient_name"] = line_val

        # ------------------------------------------------------------
        # 6) total_amount: regex first, line fallback
        # ------------------------------------------------------------
        if not result.get("total_amount"):
            best_line = get_best_line("total_amount")
            if best_line:
                line_val = _extract_amount_from_line(best_line)
                if line_val:
                    result["total_amount"] = line_val

        return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/line/extract_invoice_fields_linecls.py <file>")
        sys.exit(1)

    import json
    from run import extract_text

    file_path = sys.argv[1]
    raw_text = extract_text(file_path)
    extractor = InvoiceExtractor()
    result = extractor.extract(raw_text, file_path=file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()