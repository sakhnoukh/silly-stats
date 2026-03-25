## Phase 3 — Invoice Information Extraction

Phase 3 implements a rule-based invoice extractor over `raw_text` OCR output.
The extractor targets the six required invoice fields:

- invoice number
- invoice date
- due date
- issuer name
- recipient name
- total amount

The extractor works on `raw_text` rather than `clean_text`, because text cleaning used for classification removes useful field cues such as line structure, punctuation, and labels like `Invoice Date`, `Bill To`, or `Attn:`. The current baseline combines regex patterns with rule-based heuristics for candidate filtering and normalization. 

To make extraction more conservative on noisy OCR, the baseline also computes auxiliary signals:
- `payment_terms`
- `invoice_like_score`
- `ocr_quality_score`
- `document_confidence`

These auxiliary outputs are used to reduce false positives and to flag low-confidence documents, but the main reported fields remain the six required invoice fields. 

The baseline was evaluated on all rows with `label == "invoice"` from the processed dataset, which corresponds to 198 invoice documents across train, validation, and test splits. This gives a standalone Phase 3 evaluation under the assumption that the input document is already known to be an invoice. 

Current baseline coverage on the 198 invoice documents is:

- invoice number: 7.6%
- invoice date: 22.2%
- due date: 1.0%
- issuer name: 91.4%
- recipient name: 2.5%
- total amount: 14.1%

These results show that the baseline is able to recover some structured invoice metadata, especially issuer names and a subset of dates and totals, but extraction remains difficult due to OCR noise, layout variation, and invoice-adjacent business document formats in the archival dataset. The regex baseline is therefore treated as the reference method for later comparison against stronger extraction approaches such as template-routed extraction and candidate-ranking models. 