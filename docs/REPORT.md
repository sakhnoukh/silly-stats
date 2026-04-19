# Document Classification and Invoice Field Extraction

## IE University — Statistical Learning

### Team
Sami Akhnoukh, Sofia Bonoan, Nikolas Lafrentz, Matthew Maingot, Beatriz Martín, Salmane Mouhib

---

## Abstract

This project builds an end-to-end, non-generative document pipeline that classifies input documents into four categories — **invoice, receipt, email, and contract** — and then extracts six structured fields from invoices only: **invoice number, invoice date, due date, issuer name, recipient name, and total amount**.

The project began with a classical TF-IDF document classifier and a regex-based invoice extractor. During development, we explored several extraction strategies, including stricter regex variants, a classical ML candidate-ranker path, and a layout-aware transformer baseline. The final selected solution is a **hybrid invoice extractor** implemented in `scripts/extract_invoice_fields_v5.py`, which combines **regex/rule-based extraction**, **OCR-based rescue**, **field-specific heuristics**, and **selective NER / email-domain fallback**. This final system achieved strong results on both controlled and more realistic benchmarks, including **88.6%** on the synthetic invoice set, **90.5%** on degraded OCR invoices, and **71.2%** on a manually revised real-world invoice benchmark.

---

## 1. Problem Statement

The objective of the project was to design a practical and reproducible pipeline for scanned and digital documents under the course constraint of **no generative AI in the final solution**.

The system needed to:

1. classify a document as one of four classes:
   - invoice
   - receipt
   - email
   - contract
2. extract six key fields from invoices only:
   - `invoice_number`
   - `invoice_date`
   - `due_date`
   - `issuer_name`
   - `recipient_name`
   - `total_amount`

The main difficulty was not only handling clean synthetic invoices, but also generalizing to:
- OCR noise
- multilingual or translated invoices
- multi-column layouts
- receipt-like or weakly labeled documents
- inconsistent real-world formatting

---

## 2. End-to-End Pipeline

At a high level, the final system works as follows:

1. **Document input**: PDF, image, or text file
2. **Text extraction / OCR**:
   - PDFs are read with text extraction first
   - images are processed with Tesseract OCR
   - OCR fallback is used when needed
3. **Document classification**:
   - a TF-IDF feature representation is built from cleaned text
   - a classical ML classifier predicts the document class
4. **Invoice field extraction**:
   - if the predicted class is `invoice`, the invoice extractor runs
   - otherwise the system returns only the predicted document class
5. **Output**:
   - structured JSON through `run.py`
   - optional demo through `app.py`

This keeps the extraction stage modular and only applies invoice logic to documents predicted as invoices.

---

## 3. Phase 2 — Document Classification

The classification stage is based on **classical machine learning over TF-IDF features**. The repository includes several trained models and model-comparison artifacts. The general approach is:

- OCR/text extraction from the document
- text cleaning and normalization
- TF-IDF vectorization
- training and evaluation of several classical classifiers

The classification stage is important because it determines whether invoice extraction will run at all. One practical weakness observed during development is that invoice-like documents that resemble receipts can be misclassified upstream, which then blocks Phase 3 extraction.

---

## 4. Phase 3 — Evolution of the Invoice Extraction Approach

Invoice field extraction changed significantly during the project. The final system was not built in one step.

### 4.1 Regex baseline (v0)

The initial extractor (`v0`) was a **pure regex / rule-based baseline**.

It searched for invoice number, dates, issuer, recipient, and total directly in the extracted text. This worked reasonably on simple synthetic invoices, but it was fragile when confronted with:
- OCR artefacts
- inconsistent labels
- merged columns
- noisy real invoices

### 4.2 Regex hardening (v1–v3)

The next iterations improved the regex logic:
- stricter invoice-number patterns
- better date parsing
- better issuer vs recipient separation
- stronger amount normalization and matching

These versions made the regex system substantially stronger on the controlled synthetic invoice benchmark, but they still struggled on real-world data.

### 4.3 Exploratory classical ML ranker path

In parallel, we explored a separate **candidate-ranking ML approach**.

This path did **not** directly extract fields from the whole invoice. Instead, it worked in several steps:

1. generate candidate values for a subset of fields
2. annotate which candidates are correct
3. train one binary ranker per field
4. score candidates and choose the top one

This ML branch only targeted three fields:
- `recipient_name`
- `invoice_number`
- `total_amount`

It depended on **pre-built candidate tables**, meaning that before inference we had to generate a structured list of possible candidate values for every field/document pair. In practice, this made the approach much less convenient as an end-to-end extractor.

A further limitation is that the candidate labels used for training were produced through **LLM-assisted annotation batches**.

As a result, the ML ranker path is best described as an exploratory approach rather than the final production solution.

### 4.4 Layout-aware transformer attempt

We also explored a LayoutLMv3-based extractor. In practice, this approach did not perform well enough in our setting and was not selected as the final solution.

### 4.5 OCR-rescue hybrid extractor (v4)

The key architectural shift happened with `v4`.

Instead of relying only on flattened extracted text, `v4` introduced an **OCR rescue layer**:
- regex/rules still ran first on the text
- if a field was missing or suspicious, the extractor rasterized the page and ran Tesseract OCR
- OCR words were grouped into lines
- those OCR lines were then used as field candidates for rescue

This was the first version that clearly treated real invoice extraction as a **text + visual-layout problem**, rather than only a regex problem.

### 4.6 Final selected extractor (v5)

The final selected solution is `scripts/extract_invoice_fields_v5.py`.

`v5` kept the layered architecture introduced in `v4`, but added a strong robustness pass for real-world invoices:
- OCR cleanup
- Tesseract page segmentation tuning (`psm 6`)
- higher render scale
- European amount normalization
- multiline total detection
- stronger invoice-number anchors (e.g. booking/folio variants)
- stricter bad-value filtering for issuer and recipient
- Spanish all-caps recipient heuristic
- NER rescue for issuer / recipient
- email-domain fallback for issuer

This final version is the selected extractor because it is the strongest end-to-end practical solution in the project.

---

## 5. Final Solution in Detail (v5)

The final invoice extractor is a **layered hybrid system**.

### 5.1 First pass: regex / rule-based extraction

The extractor first runs over the raw extracted text and attempts to recover all six fields using:
- labeled regex patterns
- date normalization
- amount normalization
- issuer/recipient anchoring rules

This step is efficient and works well on clean invoices.

### 5.2 Second pass: OCR-based rescue

If the first pass fails for a field, or if a value looks suspicious, the extractor runs OCR-based recovery:
- the document page is rendered as an image
- Tesseract OCR reads the text from the image
- OCR words are grouped into lines
- OCR line candidates are used to rescue missing fields

The OCR rescue stage is particularly useful for:
- invoice numbers in visually separated headers
- totals near the bottom of the page
- issuer and recipient blocks in multi-column layouts
- real invoices where raw extracted text loses formatting

### 5.3 Third pass: heuristic hardening

After OCR rescue, additional defensive rules filter or repair bad outputs:
- reject obvious junk values
- prefer plausible totals over subtotals or line items
- recognize all-caps Spanish recipient lines
- extend invoice numbers with trailing tokens when needed
- normalize OCR-distorted outputs

### 5.4 Last-resort fallback

If issuer/recipient are still bad after regex and OCR rescue:
- spaCy NER rescue is used as a selective fallback
- a non-generic email-domain fallback can be used to recover issuer names

This layered ordering was chosen deliberately:
- regex is cheap and works well when text is clean
- OCR is more expensive and can introduce noise, so it is used only when needed
- NER and domain fallbacks are treated as last-resort recovery rather than primary extraction

---

## 6. Evaluation Strategy

The final system was evaluated on three complementary benchmarks.

### 6.1 Synthetic invoice benchmark

A controlled invoice test set under `tests/invoices/`.

Purpose:
- benchmark structured extraction in a clean setting
- compare extractor versions consistently

### 6.2 OCR-degraded benchmark

A degraded image-based benchmark under `tests/invoices_ocr/`.

Purpose:
- measure robustness when OCR quality is poor
- stress test the OCR rescue path

### 6.3 Real-world benchmark

A manually revised benchmark under:
- `tests/invoices_real/`
- `tests/ground_truth_real6.json`

Purpose:
- evaluate generalization to realistic, out-of-domain invoices
- test multilingual / translated / messy invoices
- measure performance on the type of failures that matter in practice

This real benchmark is the most important one for judging whether the extractor is genuinely useful beyond controlled templates.

---

## 7. Results

### 7.1 Final selected extractor (`v5`)

| Test set | Files | Score | Accuracy |
|---|---:|---:|---:|
| `tests/invoices/` (synthetic) | 19 | 101/114 | **88.6%** |
| `tests/invoices_ocr/` (degraded PNG) | 7 | 38/42 | **90.5%** |
| `tests/invoices_real/` (real-world) | 14 | 52/73 | **71.2%** |

Unit tests: **77/77 passing**

### 7.2 Extraction evolution summary

| Version | Synthetic | Real invoices | Key addition |
|---|---:|---:|---|
| Regex v0 | 43.3% | — | Baseline |
| Regex v3 | 85.8% | 25.6% | Stronger regex / rules |
| v4 | 90.8% | 47.1% | OCR rescue layer |
| ML ranker | 87% (RVL-CDIP-only / exploratory) | 0% | Candidate-ranker approach |
| LayoutLMv3 zero-shot | 30% | 4.7% | Training mismatch / not viable |
| **v5 (final)** | **88.6%** | **71.2%** | OCR hardening + heuristic / NER fallback |

### 7.3 Interpretation

The main project result is not that one single method was universally best. Instead, the project showed that:
- regex alone is strong on clean invoices
- OCR rescue is essential for real-world robustness
- the candidate-ranker ML path was interesting but not practical as the final end-to-end solution
- the strongest final extractor is a **hybrid layered system**, not a pure regex model and not a pure ML ranker

---

## 8. Limitations

The final system still has clear limitations.

### 8.1 Issuer and recipient remain difficult

These are the hardest fields on real invoices because they are sensitive to:
- layout variation
- multilingual labels
- merged OCR lines
- address-vs-company ambiguity

### 8.2 Due date is often absent or weakly expressed

Many real invoices do not explicitly state a due date. This limits achievable performance even with a better extractor.

### 8.3 OCR remains a major source of noise

Although OCR rescue improves performance overall, it can also introduce errors such as:
- merged columns
- corrupted company names
- bad numeric reads

### 8.4 Upstream classification can still block extraction

If an invoice-like document is misclassified as a receipt or another class, the invoice extractor is never triggered.

### 8.5 The ML ranker path is exploratory

Because the candidate-label supervision was LLM-assisted and not fully manually revised, the ML ranker branch should not be presented as fully validated supervised learning.

---

## 9. Future Improvements

The most useful next steps would be:

1. **Improve invoice-vs-receipt classification**
   - this is one of the biggest practical blockers in the full pipeline

2. **Expand the real-world gold dataset**
   - more labeled real invoices would give a more trustworthy benchmark
   - especially for multilingual, two-column, and receipt-like cases

3. **Build stronger OCR candidate ranking for hard fields**
   - especially issuer, recipient, and total amount
   - this could combine OCR line candidates with better scoring features

4. **Revisit the FATURA line-based pipeline with manually verified supervision**
   - the current line-based idea is promising, but the supervision needs to be stronger and better audited

5. **Try fine-tuned discriminative layout-aware models**
   - e.g. LayoutLMv3 / LiLT / similar token- or line-classification approaches
   - but only as a trained discriminative system, not as zero-shot plug-in inference

6. **Improve OCR preprocessing**
   - denoising, region-based OCR, multilingual OCR settings, and better page segmentation could reduce downstream extraction errors

7. **Clean and simplify the repository structure**
   - separate final code, historical baselines, and experimental branches more clearly
   - make the selected runtime path easier to understand

---

## 10. Reproducibility

Important scripts for reproducing the final workflow:

- `run.py` — unified pipeline entry point
- `app.py` — Streamlit demo
- `scripts/evaluate_pipeline.py` — benchmark extractor versions
- `scripts/show_eval_failures.py` — inspect per-field and per-file failures
- `scripts/extract_invoice_fields_v5.py` — final selected extractor
- `docs/v5_improvements.md` — detailed v5 change log and benchmark summary

Experimental / historical extraction paths retained in the repository:
- `scripts/extract_invoice_fields_v0.py` to `v4.py`
- `scripts/extract_invoice_fields_layoutlmv3.py`
- `scripts/line/` FATURA line-classification pipeline
- `scripts/phase_3_extraction/` candidate-ranking ML pipeline

---

## 11. Conclusion

The project successfully delivered a **non-generative, end-to-end document classification and invoice field extraction system**.

The main technical lesson from Phase 3 is that real-world invoice extraction cannot be solved reliably with flattened-text regex alone. The final selected solution, `v5`, therefore combines:
- regex / rules for strong first-pass extraction
- OCR-based rescue for layout-sensitive failures
- heuristic hardening for noisy real documents
- selective NER / email-domain fallback for the hardest issuer/recipient cases

This final hybrid approach achieved the best practical balance between controlled-benchmark performance and real-world robustness, making it the strongest submission-ready solution in the repository.

