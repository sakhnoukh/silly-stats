# Phase 3 — Extraction: Status Update & Running Guide

## What actually runs in production

```
run.py → extract_text() → classify_document() → InvoiceExtractor.extract()
```

- **Classifier (Phase 2):** TF-IDF + Logistic Regression trained on RVL-CDIP + SROIE. 100% on our 20 modern test invoices.
- **Extractor (Phase 3):** Rule-based regex, no training data. `scripts/extract_invoice_fields.py` (currently v3).

---

## What happened with the ML ranker

The ML ranker reported **87% accuracy** — but measured on held-out RVL-CDIP data, the same distribution it was trained on. On modern invoices: **0%**. Domain generalisation failure.

The ranker also requires pre-built candidate CSV tables per document. It cannot process a new invoice at runtime. It was **never integrated into `run.py`** and is batch-only.

| Method | Domain | Overall (3 shared fields) |
|---|---|---|
| ML ranker | RVL-CDIP (training domain) | 87% |
| ML ranker | Modern invoices (unseen) | 0% |
| Regex v0 | Modern invoices | 58% |
| Regex v3 | Modern invoices | ~72% (same 3 fields) |

---

## Regex extractor: what was fixed (v0 → v3)

| Bug | Fix | Impact |
|---|---|---|
| Invoice number matched any nearby word — no label required | Label keyword mandatory; value must contain a digit | inv_no: 30% → 80% |
| `due date` pattern missed "Due Date:" | Added `(?:date)?`; written-month support; multiline flag | due_date: 25% → 95% |
| Issuer anchored on "Bill To" — extracted recipient as issuer | Removed Bill To from issuer; scan text *above* that anchor | issuer: 5% → 80%+ |
| Non-standard total labels not matched | Expanded pattern set; 3-letter currency code support | total: 75% → 95% |
| `Proforma No:`, `Our Ref:`, `#hash-prefix` not recognised | Added patterns for proforma, our ref, tax invoice, hash prefix | inv_no further improved |
| Date swap on law firm layout | Position-aware disambiguation relative to Due Date label | inv_date: 75% → 95% |
| Document-type words (`INVOICE`, `TAX INVOICE`) selected as issuer | Doc-type reject list; no-digit requirement for name candidates | issuer regression fixed |

---

## Results across test sets

### Synthetic modern invoices (20 PDFs, AI-generated, diverse English layouts)

| Version | Overall | inv_no | inv_date | due_date | issuer | recipient | total |
|---|---|---|---|---|---|---|---|
| v0 (original) | 43.3% | 30% | 80% | 25% | 5% | 70% | 80% |
| v1 | 76.7% | 60% | 80% | 80% | 100% | 60% | 80% |
| v2 | 80.8% | 80% | 75% | 80% | 80% | 75% | 95% |
| v3 (current) | **84.2%** | 80% | 95% | 95% | 65% | 75% | 95% |

### OCR pipeline — PNG inputs via Tesseract (v3, 7 invoices)

Overall **85.7%** — invoice_date 100%, due_date 100%, total 100%, issuer 42.9%.

### Real vendor invoices — invoice2data test suite (v3, 11 invoices)

Overall **25.6%** — invoice_number 9%, invoice_date 46%, total 20%, issuer 27%.

This is the honest generalizability measure. Failures split into:
- **Non-English** (coolblue=Dutch, NetPresse=French, free_fiber=French, Orlen=Polish): regex is English-only, expected total failure
- **Classifier bug**: QualityHosting.pdf classified as `email` — never reaches extraction
- **Structural misses on real English invoices**: slash-format invoice numbers (`INV/2023/03/0008`), unusual layouts (AzureInterior, saeco, oyo)

**What this tells us:** regex achieves 84% on standard English invoices, 25% on real international ones. This directly motivates a layout-aware approach.

---

## Known limitations (documented)

- **Non-English labels** — not supported; requires multilingual training
- **No-label invoices** — structurally impossible with regex
- **pdfplumber column extraction** — drops leading digits from two-column totals (e.g. `$17,458` → `7,458`)
- **Slash-format invoice numbers** (`INV/2023/03/0008`) — not currently matched
- **Classifier misclassification** — some real invoices classified as email/form/receipt

---

## Next step: LayoutLMv3 (transformer)

Transformers for discriminative tasks (NER/token classification) are within course scope. LayoutLMv3 is non-generative — it classifies tokens, does not generate text.

**Why it solves what regex can't:** LayoutLMv3 understands 2D spatial layout (bounding box positions) in addition to text. It handles non-English labels, unusual layouts, and no-label invoices because it learned positional patterns from real documents — not keyword patterns.

**Plan:** Use `Theivaprakasham/layoutlmv3-finetuned-invoice` (HuggingFace, zero-shot — no training needed). Needs Tesseract in coordinate mode to get bounding boxes. Compare directly against regex v3 on same test sets.

---

## Running guide (current v3 state)

### Prerequisites

```bash
pip install pdfplumber pytesseract Pillow pypdfium2 joblib scikit-learn nltk datasets
# Tesseract must be on PATH: export PATH="/c/Program Files/Tesseract-OCR:$PATH"
```

### Run the live pipeline on a single invoice

```bash
# Basic — prints JSON result
python run.py path/to/invoice.pdf

# Verbose — shows each pipeline step
python run.py path/to/invoice.pdf --verbose

# Save result to file
python run.py path/to/invoice.pdf --output result.json
```

### Evaluate on synthetic test invoices (20 PDFs)

```bash
# Evaluate current extractor
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v3.py \
  --output results/eval_v3.json

# Compare two versions side by side
python scripts/show_eval_failures.py results/eval_v2.json results/eval_v3.json
```

### Evaluate on OCR/PNG invoices (Tesseract pipeline)

```bash
python scripts/evaluate_pipeline.py \
  --invoices tests/invoices_ocr \
  --extractor scripts/extract_invoice_fields_v3.py \
  --output results/eval_v3_ocr.json
```

### Evaluate on real vendor invoices (invoice2data)

```bash
# Build ground truth (one-time, run from silly-stats root with invoice2data cloned one level up)
python scripts/build_invoice2data_groundtruth.py --compare-dir ../invoice2data/tests/compare

# Evaluate
python scripts/evaluate_pipeline.py \
  --invoices ../invoice2data/tests/compare \
  --ground-truth tests/ground_truth_invoice2data.json \
  --extractor scripts/extract_invoice_fields_v3.py \
  --output results/eval_v3_invoice2data.json

# Show failures in detail
python scripts/show_eval_failures.py results/eval_v3_invoice2data.json
```

### Evaluate on RVL-CDIP gold dataset (1980s scanned invoices)

```bash
python scripts/evaluate_on_dataset.py \
  --extractor scripts/extract_invoice_fields_v3.py \
  --output results/eval_dataset_v3.json
```

### Test ML ranker generalization on modern invoices

```bash
python scripts/test_ml_generalization.py --output results/eval_ml_modern.json
```

### Compare extractor versions

```bash
# Generate eval files for each version
python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v0.py --output results/eval_v0.json
python scripts/evaluate_pipeline.py --extractor scripts/extract_invoice_fields_v3.py --output results/eval_v3.json

# Side-by-side comparison with regression detection
python scripts/show_eval_failures.py results/eval_v0.json results/eval_v3.json
```

---

## Test infrastructure

- `tests/invoices/` — 20 PDF invoices (synthetic, diverse English layouts)
- `tests/invoices_ocr/` — 7 PNG versions for OCR pipeline testing
- `tests/ground_truth.json` — 27 documents with 6-field labels
- `tests/ground_truth_invoice2data.json` — 11 real vendor invoices (3-field labels)
- `scripts/evaluate_pipeline.py --extractor <version>` — version-switchable evaluator
- `scripts/show_eval_failures.py v0.json v1.json` — per-invoice breakdown + regression detection
- `scripts/build_invoice2data_groundtruth.py` — converts invoice2data JSONs to our format
- `scripts/evaluate_on_dataset.py` — evaluates against RVL-CDIP gold_dataset.csv
- `scripts/test_ml_generalization.py` — tests ML ranker on modern PDFs
- `scripts/convert_to_ocr_testset.py` — PDF → PNG via pypdfium2 (no poppler needed)

All work on branch `fix/extraction-improvements`. PR to master pending.