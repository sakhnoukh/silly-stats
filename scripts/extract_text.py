"""
Text Extraction Pipeline
Extracts text from raw documents (PDF, image, text) and outputs a CSV with metadata.
Output: data/extracted/dataset_raw.csv
"""

import os
import csv
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")
EXTRACTED_DIR = os.path.join(PROJ_ROOT, "data", "extracted")
OUTPUT_CSV = os.path.join(PROJ_ROOT, "data", "extracted", "dataset_raw.csv")

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), "direct_text"


def extract_from_pdf(filepath):
    """Try pdfplumber first; fall back to pdfminer; then OCR."""
    text = ""
    method = "pdf_text"

    # Attempt 1: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
    except Exception:
        text = ""

    # Attempt 2: pdfminer
    if len(text) < 50:
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(filepath).strip()
        except Exception:
            text = ""

    # Attempt 3: OCR fallback
    if len(text) < 50:
        text = _ocr_pdf(filepath)
        method = "ocr"

    return text, method


def _ocr_pdf(filepath):
    """Convert PDF pages to images and run OCR."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(filepath, dpi=300)
        pages = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"    OCR failed for {filepath}: {e}")
        return ""


def extract_from_image(filepath):
    """Run OCR on an image file."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img).strip()
        return text, "ocr"
    except Exception as e:
        print(f"    OCR failed for {filepath}: {e}")
        return "", "ocr_failed"


EXTRACTORS = {
    ".txt": extract_from_txt,
    ".pdf": extract_from_pdf,
    ".png": extract_from_image,
    ".jpg": extract_from_image,
    ".jpeg": extract_from_image,
}

# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction():
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    rows = []
    doc_counter = 0

    labels = sorted([d for d in os.listdir(RAW_DIR)
                     if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith(".")])

    for label in labels:
        label_dir = os.path.join(RAW_DIR, label)
        files = sorted(os.listdir(label_dir))
        print(f"[{label}] found {len(files)} files")

        for fname in files:
            fpath = os.path.join(label_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext not in SUPPORTED_EXTENSIONS:
                print(f"  SKIP unsupported: {fname}")
                continue

            extractor = EXTRACTORS[ext]
            text, method = extractor(fpath)

            doc_counter += 1
            doc_id = f"doc_{doc_counter:04d}"

            # Save extracted text to extracted/ mirror
            out_dir = os.path.join(EXTRACTED_DIR, label)
            os.makedirs(out_dir, exist_ok=True)
            txt_fname = os.path.splitext(fname)[0] + ".txt"
            txt_path = os.path.join(out_dir, txt_fname)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            rows.append({
                "doc_id": doc_id,
                "file_name": fname,
                "label": label,
                "source_dataset": "pilot_synthetic",
                "file_type": ext,
                "extraction_method": method,
                "raw_text": text,
            })

            status = "OK" if len(text) > 0 else "EMPTY"
            print(f"  {doc_id} | {fname} | {method} | {len(text)} chars | {status}")

    # Write CSV
    fieldnames = ["doc_id", "file_name", "label", "source_dataset",
                  "file_type", "extraction_method", "raw_text"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtraction complete: {len(rows)} documents -> {OUTPUT_CSV}")
    return rows


if __name__ == "__main__":
    run_extraction()
