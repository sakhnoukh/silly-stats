# Document Classification & Invoice Extraction System

A complete **non-generative** document pipeline that classifies documents into 4 categories (**Invoice**, **Receipt**, **Email**, **Contract**) and extracts structured fields from invoices using **classical ML, OCR, rule-based extraction, and discriminative NLP/NER rescue**.

---

## Project Overview

### Goal

Build an end-to-end system that:

1. Classifies documents into 4 document types
2. Extracts 6 key fields **only from invoices**:
   - invoice number
   - invoice date
   - due date
   - issuer name
   - recipient name
   - total amount

### Constraints

- No generative AI
- Classical / discriminative methods only
- Handle OCR noise, multilingual wording, and varied invoice layouts
- Support both PDF and image inputs

---

## Current Selected Version

### Selected extractor

**`scripts/extract_invoice_fields_v5.py`**

This is the current best Phase 3 extractor and the version used by the integrated pipeline.

### Current extraction approach

Phase 3 is a **hybrid extractor**, not just regex:

- **Regex / rule-based core extraction**
- **OCR-based rescue** using Tesseract on the document image
- **OCR cleanup + layout-aware heuristics**
- **Field-specific recovery logic** for invoice number, dates, issuer, recipient, and total
- **Selective spaCy NER rescue** for difficult issuer / recipient cases (XLM-RoBERTa multilingual for ORG, RoBERTa English transformer for PERSON — both discriminative encoder-only models)
- **Post-processing and bad-value filtering** for noisy real-world documents

Historical extractor versions (`v0`–`v4`) are kept in the repo as baselines / checkpoints. The FATURA line-classification pipeline is also kept as an experimental track, but is **not** the selected production extractor.

---

## Project Status

| Phase | Description | Owner(s) | Status |
|---|---|---|---|
| **Phase 1** | Data Collection & Preprocessing | Sami | ✅ Complete |
| **Phase 2** | Document Classification | Sofia | ✅ Complete |
| **Phase 3** | Invoice Field Extraction | Bea & Salmane | ✅ Complete |
| **Phase 4** | Pipeline Integration & Demo | Matthew | ✅ Complete |
| **Phase 5** | Documentation & Presentation | Niko | ✅ Complete |

---

## Architecture

```
Input Document (PDF/Image)
  ↓
[Phase 1] OCR / Text Extraction
  ↓
[Phase 2] Document Classification (TF-IDF + Logistic Regression)
  ↓
  ├─ If NOT invoice → return document class only
  └─ If invoice → [Phase 3] Field Extraction (v5)
                   = Regex + OCR rescue + heuristic/NER fallback
  ↓
[Phase 4] Output Results (JSON / Streamlit UI)
```

---

## Current Results

### Phase 2 — Classification

- Best classifier: Logistic Regression
- TF-IDF based multi-class document classification
- Used as the routing step before invoice extraction

### Phase 3 — Selected extractor (v5)

| Test set | Files | Score | Accuracy |
|---|---|---|---|
| `tests/invoices/` (synthetic) | 19 | 101/114 | 88.6% |
| `tests/invoices_ocr/` (degraded PNG) | 7 | 38/42 | 90.5% |
| `tests/invoices_real/` (real-world) | 14 | 52/73 | 71.2% |

Unit tests: 77/77 passing

---

## Quick Start

### Requirements

- Python 3.10 recommended
- Tesseract OCR installed
- `pip install -r requirements.txt`

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# OR
.venv\Scripts\activate             # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Install Tesseract

**macOS**
```bash
brew install tesseract
```

**Ubuntu / Debian**
```bash
sudo apt-get update && sudo apt-get install tesseract-ocr
```

**Windows** — install Tesseract OCR separately and ensure it is on PATH (or use the Windows-specific path configuration already handled in `extract_invoice_fields_v5.py`).

### Install spaCy NER models (required for v5)

```bash
python -m spacy download xx_ent_wiki_sm    # multilingual — issuer rescue
python -m spacy download en_core_web_trf   # English transformer — recipient rescue
```

### Run the complete pipeline

**Option 1 — Streamlit demo**
```bash
streamlit run app.py
```

**Option 2 — CLI**
```bash
python run.py --input path/to/document.pdf --output results.json
```

This will extract text, classify the document, and run invoice extraction only when the predicted class is `invoice`.

### Run the selected extractor directly

```bash
python scripts/extract_invoice_fields_v5.py "Invoice No: INV-001 Total: 99.00"
# or from a text file:
python scripts/extract_invoice_fields_v5.py path/to/invoice_text.txt
```

---

## Evaluation Commands

```bash
# Synthetic invoice benchmark
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --invoices tests/invoices \
  --ground-truth tests/ground_truth.json \
  --output results/eval_v5_synthetic.json

# Degraded OCR benchmark
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --invoices tests/invoices_ocr \
  --ground-truth tests/ground_truth_invoice2data.json \
  --output results/eval_v5_ocr.json

# Real-world benchmark (main generalization test)
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --invoices tests/invoices_real \
  --ground-truth tests/ground_truth_real6.json \
  --output results/eval_v5_real.json

# Show detailed per-invoice failures
python scripts/show_eval_failures.py results/eval_v5_real.json

# Compare two versions
python scripts/show_eval_failures.py results/eval_v4_real.json results/eval_v5_real.json

# RVL-CDIP gold dataset (secondary stress test)
python scripts/evaluate_on_dataset.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --output results/eval_dataset_v5.json

# Unit tests
python -m pytest tests/test_phase3.py -v
```

---

## Technical Approach

### Phase 1 — Data Collection & Preprocessing

- OCR extraction from PDFs and images
- Text cleaning and preprocessing
- Dataset preparation and TF-IDF feature building

### Phase 2 — Document Classification

Multi-class classification into: Invoice, Receipt, Email, Contract.

Best-performing model: Logistic Regression with TF-IDF features. Other classical baselines also trained and compared.

### Phase 3 — Invoice Information Extraction

The selected extractor (v5) combines three layers:

**1) Rule-based extraction**
- Regex patterns for invoice number, dates, names, and totals
- Normalisation of dates (→ YYYY-MM-DD) and amounts (→ decimal string)
- Support for multiple invoice styles and label variants

**2) OCR-based rescue**

When raw text extraction is weak or misleading, v5:
- Rasterizes the document page with pypdfium2
- Runs Tesseract in word-coordinate mode
- Groups OCR words into column-aware lines
- Applies field-specific heuristics to recover missing or bad values using position and keyword anchors

**3) NER rescue**

Last resort for issuer and recipient when both regex and OCR fail:
- `xx_ent_wiki_sm` (XLM-RoBERTa multilingual) for ORG entities → issuer
- `en_core_web_trf` (RoBERTa English transformer) for PERSON entities → recipient
- Both are discriminative encoder-only models; no generative AI
- NER only fires when the current value is flagged as bad by internal filters

### Phase 4 — Integration

- Unified `run.py` entry point with JSON output
- Streamlit demo in `app.py`
- Error handling across all pipeline stages

---

## Project Structure

```
silly-stats/
├── README.md
├── CONTRIBUTORS.md
├── CURRENT_STATUS.md
├── app.py
├── run.py
├── requirements.txt
│
├── data/
│   ├── dataset_manifest.json
│   └── features/
│
├── docs/
│   ├── data_modeling.md
│   ├── dataset_report.md
│   ├── exploration_findings.md
│   └── v5_improvements.md
│
├── models/
│   ├── best_model.pkl
│   ├── best_model_meta.json
│   └── phase3_extraction/
│
├── results/
│   ├── model_comparison.csv
│   ├── extraction_coverage.txt
│   ├── invoice_extractions.json
│   └── phase3_extraction/
│
├── scripts/
│   ├── train_models.py
│   ├── evaluate_model.py
│   ├── evaluate_pipeline.py
│   ├── evaluate_on_dataset.py
│   ├── show_eval_failures.py
│   ├── quick_eval.py
│   ├── quick_eval_synthetic.py
│   ├── extract_invoice_fields_v0.py   ← historical baseline
│   ├── extract_invoice_fields_v1.py
│   ├── extract_invoice_fields_v2.py
│   ├── extract_invoice_fields_v3.py
│   ├── extract_invoice_fields_v4.py
│   ├── extract_invoice_fields_v5.py   ← selected production extractor
│   └── line/
│       ├── build_fatura_line_dataset.py
│       ├── train_line_rankers.py
│       └── extract_invoice_fields_linecls.py
│
└── tests/
    ├── invoices/
    ├── invoices_ocr/
    ├── invoices_real/
    ├── ground_truth.json
    ├── ground_truth_invoice2data.json
    ├── ground_truth_real6.json
    └── test_phase3.py
```

---

## Important Notes

### Selected vs historical extractors

- Use `extract_invoice_fields_v5.py` as the default extractor
- `v0`–`v4` are preserved as historical baselines and iteration checkpoints
- `scripts/line/` contains the FATURA line-based experimental pipeline (explored approach, not production)

### Real-world benchmark

`tests/invoices_real/` + `tests/ground_truth_real6.json` is the main out-of-domain robustness set used to track practical generalization. It contains real English and Spanish-translated-to-English invoices.

### Current limitations

Remaining weak spots in real-world invoices:

- **Due date (20%)** — field is genuinely absent in most real invoices in this set
- **Issuer on noisy/translated documents** — OCR font misreads (OUIGO O→Q, Carrefour truncation)
- **Subtotal vs final total confusion** on some receipts and unlabeled layouts
- **Upstream misclassification** — some invoice-like documents classified as receipt by Phase 2, blocking extraction entirely
- **Spanish last-first name format** — not reliably recognized as PERSON by English NER model

Full failure analysis in `CURRENT_STATUS.md`.

---

## Team & Contributions

See `CONTRIBUTORS.md` for phase assignments and contribution details.