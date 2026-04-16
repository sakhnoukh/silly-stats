# Phase 3 — Invoice Field Extraction: Current Status

## Selected extractor: `scripts/extract_invoice_fields_v5.py`

---

## Pipeline

```
run.py → extract_text() → classify_document() → InvoiceExtractor.extract()
```

**Phase 2 classifier:** TF-IDF + Logistic Regression. Works well on standard invoices; occasionally misclassifies receipt-like documents (e.g. RENTA tax form classified as receipt, blocking extraction entirely).

**Phase 3 extractor (v5):** Three-layer hybrid — no generative AI used at any stage.

1. **Regex/rule-based layer** — runs first over flat pdfplumber-extracted text. Covers standard English invoice layouts with explicit field labels.
2. **OCR rescue layer** — when regex misses or returns weak values, rasterizes page 0 with pypdfium2, runs Tesseract in word-coordinate mode, groups words into column-aware OCR lines, applies field-specific rescue logic using position and keyword anchors.
3. **NER rescue layer** — last resort when both regex and OCR fail. Uses two spaCy models: `xx_ent_wiki_sm` (XLM-RoBERTa multilingual, for ORG entities → issuer) and `en_core_web_trf` (RoBERTa English transformer, for PERSON entities → recipient). Both are discriminative encoder-only models, not generative AI. NER only fires when the current value is flagged as bad by `_looks_bad_issuer` / `_looks_bad_recipient`.

---

## Results

### 1. Synthetic modern invoices — `tests/invoices/` (19 scored PDFs)

| Field | Accuracy |
|---|---|
| invoice_number | 100.0% |
| invoice_date | 94.7% |
| due_date | 94.7% |
| issuer_name | 84.2% |
| recipient_name | 73.7% |
| total_amount | 94.7% |
| **Overall** | **90.4%** (103/114) |

Note: `invoice_11_rental.pdf` produced an EOF error and was excluded from scoring.

### 2. Real + translated invoices — `tests/invoices_real/` (14 files — main generalization benchmark)

| Field | Accuracy |
|---|---|
| invoice_number | 76.9% |
| invoice_date | 92.3% |
| due_date | 20.0% |
| issuer_name | 61.5% |
| recipient_name | 54.5% |
| total_amount | 61.5% |
| **Overall** | **66.2%** (45/68) |

---

## Version progression

| Version | Synthetic | Real invoices | Key addition |
|---|---|---|---|
| Regex v0 | 43.3% | — | Baseline |
| Regex v3 | 85.8% | 25.6% | Fixed patterns, English only |
| v4 | 90.8% | 47.1% | + OCR rescue layer |
| **v5 (current)** | **90.4%** | **66.2%** | + NER rescue layer |
| ML ranker | 87% (RVL-CDIP only) | 0% | Domain-locked, not integrated |
| LayoutLMv3 zero-shot | 30% | 4.7% | Training mismatch, not viable |

---

## Failure analysis — real invoices

Remaining failures decompose into four categories. No further regex/OCR/NER improvement addresses these without either training data or better OCR.

**Structurally absent fields (not fixable without training data):**
- `due_date` 20% — most real invoices in this set don't contain an explicit due date label. The field is genuinely absent, not mis-extracted.
- OYO total — receipt-style hotel booking with no standard total label.

**OCR read errors (fixable only with better OCR engine):**
- OUIGO issuer: `QUIGO ESPANA, SAU` — OCR misreads O→Q. The extraction logic is correct; Tesseract fails on this font.
- Carrefour issuer: `Carrefour (®` — OCR truncates at trademark symbol.
- Neony total: `87186.00` instead of `1871.86` — OCR digit transposition.

**Ground truth ambiguity:**
- OYO issuer: extractor returns `Oravel Stays Pvt. Ltd.` (registered legal entity); ground truth expects `OYO` (brand name).
- Neony issuer: extractor returns `NEONY` (brand); ground truth expects `BEATRIZ MARTIN MARTIN` (owner name).

**Hard NLP cases (would require fine-tuning):**
- Spanish last-first name format: `MARTIN MARTIN BEATRIZ` not recognized as PERSON by English NER.
- Seur recipient: NER returns `SEGOVIA` (a city/GPE) instead of customer name.
- Leroy Merlin recipient: OCR produces `ORIGINAL` (watermark artefact) that passes all filters.

---

## Known limitations

- **Due date (20% on real)** — most real invoices in this test set do not contain an explicit due date. This is not an extractor failure; the field is absent.
- **Classifier misclassification** — RENTA tax form classified as receipt, scores 0 fields. Requires adding hard negatives to Phase 2 training data.
- **OCR quality ceiling** — Tesseract misreads fonts on some translated PDFs. Swapping to DocTR or PaddleOCR would help.
- **NER language gap** — `en_core_web_trf` does not reliably detect Spanish last-first name format as PERSON. Fine-tuning on ~15 annotated examples would fix this.
- **Real evaluation set is small** — 14 invoices. Per-field percentages are sensitive to individual document failures.

---

## Required packages

```
pdfplumber
pytesseract          # Tesseract must be on PATH
Pillow
pypdfium2
scikit-learn
joblib
nltk

# NER layer
spacy
# models (run once):
#   python -m spacy download xx_ent_wiki_sm    # multilingual (issuer)
#   python -m spacy download en_core_web_trf   # English transformer (recipient)
```

---

## Running guide

```bash
# Live pipeline on a single invoice
python run.py path/to/invoice.pdf

# Evaluate on synthetic invoices
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --invoices tests/invoices \
  --ground-truth tests/ground_truth.json \
  --output results/eval_v5_synthetic.json

# Evaluate on real + translated invoices (main benchmark)
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v5.py \
  --invoices tests/invoices_real \
  --ground-truth tests/ground_truth_real6.json \
  --output results/eval_v5_real.json

# Show detailed per-invoice failures
python scripts/show_eval_failures.py results/eval_v5_real.json

# Compare two versions
python scripts/show_eval_failures.py results/eval_v4_real.json results/eval_v5_real.json
```

---

## What would be needed to go further

| Target | What it requires | Estimated effort |
|---|---|---|
| Fix Spanish name recipient (OUIGO, Seur) | Fine-tune spaCy NER on ~15 annotated name examples | 2–3 hours, no GPU |
| Fix OCR read errors (OUIGO font, Carrefour) | Swap Tesseract → DocTR or PaddleOCR | Half day |
| Recover RENTA (classifier) | Add tax-form hard negatives to Phase 2 training | 1 hour |
| Reach 75%+ on real invoices | Fine-tune LayoutLMv3 on annotated bounding-box data | Full day, GPU needed |
