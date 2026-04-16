#!/usr/bin/env python3
"""
Document Classification & Invoice Extraction Pipeline
Phase 4 — End-to-end integration

Accepts a document (image, PDF, or text file), classifies it into one of
four categories (email, form, invoice, receipt), and extracts structured
fields if the document is an invoice.

Usage:
    python3 run.py <file_path>
    python3 run.py <file_path> --output result.json
    python3 run.py <file_path> --verbose

Examples:
    python3 run.py sample_invoice.png
    python3 run.py document.pdf --output result.json
    python3 run.py invoice.txt --verbose
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJ_ROOT / "models"
FEATURES_DIR = PROJ_ROOT / "data" / "features"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = FEATURES_DIR / "tfidf_vectorizer.pkl"

# Label mapping (alphabetical — matches sklearn LabelEncoder on our 4 classes)
LABEL_MAP = {0: "email", 1: "form", 2: "invoice", 3: "receipt"}

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".csv", ".md"}


# ---------------------------------------------------------------------------
# Step 1 — Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_image(file_path: str) -> str:
    """Run Tesseract OCR on an image file and return extracted text."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("ERROR: pytesseract and Pillow are required for image OCR.")
        print("  Install: pip install pytesseract Pillow")
        print("  Also install Tesseract OCR: brew install tesseract (macOS)")
        sys.exit(1)

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using pdfplumber, falling back to OCR."""
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber is required for PDF text extraction.")
        print("  Install: pip install pdfplumber")
        sys.exit(1)

    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    text = "\n".join(text_parts)

    # If pdfplumber got nothing (scanned PDF), try OCR
    if not text.strip():
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(file_path)
            ocr_parts = []
            for img in images:
                ocr_parts.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_parts)
        except ImportError:
            print("WARNING: PDF has no extractable text and pdf2image/pytesseract")
            print("  are not installed for OCR fallback.")
            print("  Install: pip install pdf2image pytesseract")

    return text


def extract_text_from_file(file_path: str) -> str:
    """Read plain text from a text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """
    Detect file type and extract text using the appropriate method.
    Returns the raw extracted text.
    """
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(file_path)
    elif ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(file_path)
    elif ext in TEXT_EXTENSIONS:
        return extract_text_from_file(file_path)
    else:
        # Try reading as text as a last resort
        try:
            return extract_text_from_file(file_path)
        except Exception:
            print(f"ERROR: Unsupported file type '{ext}'.")
            print(f"  Supported: {sorted(IMAGE_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS)}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 — Text Cleaning (reuses Phase 1 logic)
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Clean raw text for classification. Mirrors scripts/clean_text.py logic.
    Keeps digits, currency symbols, and email-like patterns.
    """
    # Try to load NLTK resources; fall back to simple tokenization if unavailable
    use_nltk = False
    stop_words = set()

    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        stop_words = set(stopwords.words("english"))
        use_nltk = True
    except (ImportError, LookupError):
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            stop_words = set(stopwords.words("english"))
            use_nltk = True
        except Exception:
            # Minimal fallback stopwords if NLTK is completely unavailable
            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "of", "in", "to", "for", "with", "on", "at", "from", "by",
                "and", "or", "but", "not", "no", "if", "as", "it", "its",
                "this", "that", "these", "those", "i", "you", "he", "she",
                "we", "they", "me", "him", "her", "us", "them", "my", "your",
            }

    text = raw

    # Normalize unicode whitespace
    text = re.sub(r"[\xa0\u200b\u2028\u2029]", " ", text)

    # Lowercase
    text = text.lower()

    # Replace newlines with spaces
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove long repeated character sequences (OCR artifacts)
    text = re.sub(r"(.)\1{4,}", r"\1\1", text)

    # Tokenize
    if use_nltk:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()

    # Remove stopwords, keep alphanumeric and currency tokens
    cleaned_tokens = []
    for t in tokens:
        if t in stop_words:
            continue
        if re.search(r"[a-z0-9$€£¥]", t):
            cleaned_tokens.append(t)

    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# Step 3 — Classification
# ---------------------------------------------------------------------------

def load_classifier():
    """
    Load the trained model and TF-IDF vectorizer.
    Returns (model, vectorizer) or (None, None) if files are missing.
    """
    if not BEST_MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None

    import joblib
    model = joblib.load(BEST_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def classify_with_model(cleaned_text: str, model, vectorizer) -> dict:
    """
    Classify using the trained ML model + TF-IDF vectorizer.
    """
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    label = LABEL_MAP[prediction]

    # Get confidence scores if the model supports predict_proba
    confidence = {}
    try:
        probas = model.predict_proba(features)[0]
        for idx, prob in enumerate(probas):
            confidence[LABEL_MAP[idx]] = round(float(prob), 4)
    except AttributeError:
        confidence = {label: 1.0}

    return {
        "predicted_class": label,
        "confidence": confidence,
        "method": "ml_model",
    }


def classify_with_keywords(raw_text: str) -> dict:
    """
    Fallback keyword-based classifier when the ML model/vectorizer is unavailable.
    Uses simple heuristics based on document structure and vocabulary.
    """
    text_lower = raw_text.lower()

    # Score each category based on keyword presence
    scores = {"email": 0.0, "form": 0.0, "invoice": 0.0, "receipt": 0.0}

    # --- Email signals ---
    email_keywords = [
        (r"\b(from|to|cc|bcc)\s*:", 3),
        (r"\bsubject\s*:", 4),
        (r"\b(dear|hi|hello|regards|sincerely)\b", 2),
        (r"\b(re:|fw:|fwd:)", 3),
        (r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", 4),
        (r"\b(meeting|schedule|discuss|attached|please)\b", 1),
    ]

    # --- Invoice signals ---
    invoice_keywords = [
        (r"\binvoice\b", 5),
        (r"\binvoice\s*(no|number|#|id)\b", 5),
        (r"\b(bill\s*to|sold\s*to|ship\s*to)\b", 4),
        (r"\b(due\s*date|payment\s*terms|net\s*\d+)\b", 3),
        (r"\b(subtotal|total\s*amount|amount\s*due|balance\s*due)\b", 4),
        (r"\b(remit|payable)\b", 3),
        (r"\bpurchase\s*order\b", 2),
    ]

    # --- Receipt signals ---
    receipt_keywords = [
        (r"\b(receipt|cashier|store\s*#|transaction)\b", 4),
        (r"\b(subtotal|tax|total)\b", 2),
        (r"\b(visa|mastercard|cash|change|approved)\b", 4),
        (r"\b(thank\s*you\s*for\s*(shopping|your\s*purchase))\b", 5),
        (r"\b(qty|quantity)\b", 2),
        (r"\d{2}[:/]\d{2}[:/]\d{2,4}\s+\d{2}:\d{2}", 3),  # date + time pattern
    ]

    # --- Form signals ---
    form_keywords = [
        (r"\b(form|application|registration|enrollment)\b", 3),
        (r"\b(please\s*(fill|complete|check|sign))\b", 4),
        (r"\b(signature|date\s*of\s*birth|social\s*security)\b", 3),
        (r"\b(checkbox|check\s*one|mark\s*one)\b", 4),
        (r"\b(section\s*[a-z0-9]|part\s*[a-z0-9])\b", 2),
        (r"_{3,}|\[[\s]*\]", 3),  # blank lines or checkboxes
    ]

    keyword_sets = {
        "email": email_keywords,
        "invoice": invoice_keywords,
        "receipt": receipt_keywords,
        "form": form_keywords,
    }

    for category, keywords in keyword_sets.items():
        for pattern, weight in keywords:
            matches = len(re.findall(pattern, text_lower))
            scores[category] += matches * weight

    # Normalize to rough confidence
    total = sum(scores.values())
    if total == 0:
        # No signals found — default to form (generic document)
        return {
            "predicted_class": "form",
            "confidence": {"email": 0.25, "form": 0.25, "invoice": 0.25, "receipt": 0.25},
            "method": "keyword_fallback",
        }

    confidence = {k: round(v / total, 4) for k, v in scores.items()}
    predicted = max(scores, key=scores.get)

    return {
        "predicted_class": predicted,
        "confidence": confidence,
        "method": "keyword_fallback",
    }


def classify_document(raw_text: str, cleaned_text: str) -> dict:
    """
    Classify a document. Uses ML model if available, falls back to keywords.
    """
    model, vectorizer = load_classifier()

    if model is not None and vectorizer is not None:
        return classify_with_model(cleaned_text, model, vectorizer)
    else:
        return classify_with_keywords(raw_text)


# ---------------------------------------------------------------------------
# Step 4 — Invoice Field Extraction
# ---------------------------------------------------------------------------

def extract_invoice_fields(raw_text: str) -> dict:
    """
    Extract structured fields from an invoice using the Phase 3 extractor.
    Uses raw_text (not cleaned) because cleaning strips keywords like
    'Invoice Date:', 'Bill To:', etc. that the extractor relies on.
    """
    sys.path.insert(0, str(PROJ_ROOT))
    from scripts.extract_invoice_fields_v3 import InvoiceExtractor

    extractor = InvoiceExtractor()
    fields = extractor.extract(raw_text)

    return fields


# ---------------------------------------------------------------------------
# Step 5 — Build Output
# ---------------------------------------------------------------------------

def build_result(file_path: str, classification: dict, raw_text: str) -> dict:
    """
    Build the final output JSON combining classification and extraction.
    """
    result = {
        "file": os.path.basename(file_path),
        "classification": classification["predicted_class"],
        "confidence": classification["confidence"],
    }

    # Only extract fields if classified as invoice
    if classification["predicted_class"] == "invoice":
        fields = extract_invoice_fields(raw_text)
        result["extracted_fields"] = fields

        # Count how many fields were successfully extracted
        found = sum(1 for v in fields.values() if v is not None)
        result["fields_extracted"] = f"{found}/{len(fields)}"

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(file_path: str, verbose: bool = False) -> dict:
    """
    Full end-to-end pipeline: input → extract text → classify → extract fields.
    """
    # --- Validate input ---
    if not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    if verbose:
        print()
        print("=" * 60)
        print("  Document Classification & Invoice Extraction Pipeline")
        print("=" * 60)
        print()
        print(f"  Input file:  {os.path.basename(file_path)}")
        print(f"  File type:   {Path(file_path).suffix.lower()}")
        print()

    # --- Step 1: Extract text ---
    if verbose:
        print("-" * 60)
        print("  STEP 1: Text Extraction")
        print("-" * 60)

    raw_text = extract_text(file_path)

    if not raw_text or not raw_text.strip():
        if verbose:
            print("  Result:      No text could be extracted")
            print()
        return {
            "file": os.path.basename(file_path),
            "classification": "unknown",
            "confidence": {},
            "error": "No text extracted from document",
        }

    if verbose:
        word_count = len(raw_text.split())
        print(f"  Words extracted:  {word_count}")
        # Show a short preview of the text
        preview = raw_text.replace('\n', ' ').strip()[:100]
        print(f"  Preview:          {preview}...")
        print()

    # --- Step 2: Clean text for classification ---
    if verbose:
        print("-" * 60)
        print("  STEP 2: Text Cleaning")
        print("-" * 60)

    cleaned = clean_text(raw_text)

    if not cleaned.strip():
        if verbose:
            print("  Result:      Text was empty after cleaning")
            print()
        return {
            "file": os.path.basename(file_path),
            "classification": "unknown",
            "confidence": {},
            "error": "Text empty after cleaning",
        }

    if verbose:
        print(f"  Words after cleaning:  {len(cleaned.split())}")
        print()

    # --- Step 3: Classify ---
    if verbose:
        print("-" * 60)
        print("  STEP 3: Document Classification")
        print("-" * 60)

    classification = classify_document(raw_text, cleaned)

    if verbose:
        label = classification["predicted_class"]
        conf = classification["confidence"].get(label, 0)
        method = classification.get("method", "unknown")
        method_label = "ML Model (TF-IDF + Logistic Regression)" if method == "ml_model" else "Keyword-based Fallback"
        print(f"  Predicted class:  {label.upper()}")
        print(f"  Confidence:       {conf:.1%}")
        print(f"  Method:           {method_label}")
        print()
        print("  All scores:")
        for cls in ["email", "form", "invoice", "receipt"]:
            score = classification["confidence"].get(cls, 0)
            bar = "#" * int(score * 30)
            marker = " <--" if cls == label else ""
            print(f"    {cls:10s}  [{bar:<30s}] {score:.1%}{marker}")
        print()

    # --- Step 4: Extract fields if invoice ---
    result = build_result(file_path, classification, raw_text)

    if verbose and classification["predicted_class"] == "invoice":
        print("-" * 60)
        print("  STEP 4: Invoice Field Extraction")
        print("-" * 60)
        fields = result.get("extracted_fields", {})
        found = sum(1 for v in fields.values() if v is not None)
        print(f"  Fields found: {found}/{len(fields)}")
        print()
        for field, value in fields.items():
            label = field.replace("_", " ").title()
            if value:
                print(f"    {label:20s}  {value}")
            else:
                print(f"    {label:20s}  -- not found --")
        print()

    if verbose:
        print("=" * 60)
        print("  RESULT")
        print("=" * 60)
        print()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Document Classification & Invoice Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py invoice.png
  python3 run.py document.pdf --output result.json
  python3 run.py receipt.txt --verbose

Supported file types:
  Images:  .png, .jpg, .jpeg, .tif, .tiff, .bmp  (requires Tesseract OCR)
  PDFs:    .pdf  (requires pdfplumber, falls back to OCR for scanned PDFs)
  Text:    .txt, .csv, .md
        """,
    )
    parser.add_argument(
        "file",
        help="Path to the document file to classify",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save result to a JSON file",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        help="Print detailed processing information",
        action="store_true",
    )

    args = parser.parse_args()

    # Run the pipeline
    result = run_pipeline(args.file, verbose=args.verbose)

    # Output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(output_json)

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nResult saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()