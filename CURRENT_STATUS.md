# Phase 3 — Invoice Field Extraction: Current Status

## Selected extractor: `scripts/extract_invoice_fields_v4.py`

---

## Pipeline

```
run.py → extract_text() → classify_document() → InvoiceExtractor.extract()
```

**Phase 2 classifier:** TF-IDF + Logistic Regression (unchanged). Works well on controlled test set; occasionally misclassifies receipt-like real documents as receipts, blocking extraction.

**Phase 3 extractor (v4):** Hybrid — regex base layer + OCR rescue layer.
- Regex runs first over flat extracted text
- When regex misses or produces weak results, v4 rasterizes page 0, runs Tesseract, groups words into lines with column-aware splitting, and applies field-specific OCR rescue
- No generative AI

---

## Results

### 1. Synthetic modern invoices — `tests/invoices/` (20 PDFs, primary dev benchmark)

| Field | Accuracy |
|---|---|
| invoice_number | 100.0% |
| invoice_date | 95.0% |
| due_date | 95.0% |
| issuer_name | 85.0% |
| recipient_name | 75.0% |
| total_amount | 95.0% |
| **Overall** | **90.8%** (109/120) |

### 2. Real + translated invoices — `tests/invoices_real/` (14 files — main generalization benchmark)

| Field | Accuracy |
|---|---|
| invoice_number | 76.9% |
| invoice_date | 84.6% |
| due_date | 20.0% |
| issuer_name | 23.1% |
| recipient_name | 36.4% |
| total_amount | 53.8% |
| **Overall** | **52.9%** (36/68) |

### 3. RVL-CDIP gold dataset (secondary stress test — 1980s scanned invoices)

invoice_number 60%, recipient_name 8.7%, total_amount 49.5% → **Overall 35.3%**

Not the optimization target. Useful as a robustness check only.

---

## Comparison across extractor versions

| Version | Synthetic | Real invoices | Notes |
|---|---|---|---|
| Regex v0 (original) | 43.3% | — | Baseline |
| Regex v3 | 85.8% | 25.6% | English patterns only |
| v4 (current) | **90.8%** | **52.9%** | Hybrid regex + OCR rescue |
| ML ranker | 87% (RVL-CDIP only) | 0% | Domain-locked |
| LayoutLMv3 zero-shot | 30% | 4.7% | Training mismatch, not viable |

---

## FATURA line-based pipeline (experimental, not production)

Implemented under `scripts/line/`. Trains one line classifier per field on FATURA annotations.

- **Worked:** invoice_date, due_date, recipient_name, total_amount (strong held-out FATURA performance)
- **Did not work:** invoice_number, issuer_name (no usable positive labels in FATURA supervision)
- **Decision:** kept as experimental / future direction. Does not beat v4 on project benchmarks end-to-end.

---

## Known limitations

- **Issuer on real invoices (23%)** — translation banners, OCR merging, issuer/recipient swap in two-column layouts
- **Due date on real invoices (20%)** — many real invoices don't include one; multilingual phrasing not covered
- **Total on real invoices (54%)** — subtotal vs grand total confusion; translated receipts with unusual pricing
- **Classifier misclassification** — some invoice-like real documents classified as receipt, blocking extraction
- **Real evaluation set small** — 14 files, useful but not large enough to optimize against safely

---

## Running guide

```bash
# Live pipeline
python run.py path/to/invoice.pdf

# Evaluate on synthetic invoices
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v4.py \
  --invoices tests/invoices \
  --ground-truth tests/ground_truth.json \
  --output results/eval_v4_modern.json

# Evaluate on real + translated invoices
python scripts/evaluate_pipeline.py \
  --extractor scripts/extract_invoice_fields_v4.py \
  --invoices tests/invoices_real \
  --ground-truth tests/ground_truth_real6.json \
  --output results/eval_v4_real.json

# Show detailed failures
python scripts/show_eval_failures.py results/eval_v4_real.json

# Evaluate on RVL-CDIP gold
python scripts/evaluate_on_dataset.py \
  --extractor scripts/extract_invoice_fields_v4.py \
  --output results/eval_dataset_v4.json

# Train FATURA line models
python scripts/line/train_line_rankers.py
```

---

## Suggested next steps

1. **Improve Phase 2 classifier** — add receipt/invoice hard negatives to stop misclassification blocking extraction
2. **OCR candidate ranker for hard fields** — issuer, recipient, due_date need position + lexical features, classical ML ranking
3. **More real labeled invoices** — the real benchmark is too small; multilingual and two-column examples most needed
4. **FATURA line pipeline** — recover supervision for invoice_number and issuer; add confidence gating; combine selectively with v4
5. **Fine-tuned LayoutLMv3** — zero-shot failed; fine-tuning on our labeled data is the right approach if transformers are tried again
