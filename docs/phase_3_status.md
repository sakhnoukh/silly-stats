## Phase 3 — Current Status

### Goal
Extract the required invoice fields from documents classified as invoices:
- invoice number
- invoice date
- due date
- issuer name
- recipient name
- total amount

### Methods implemented
#### Method A — Regex / rule-based baseline
Uses raw OCR text with regex rules, heuristics, validation, and confidence signals.

Current coverage on 198 invoice-labeled documents:
- invoice number: 15 / 198 (7.6%)
- invoice date: 44 / 198 (22.2%)
- due date: 2 / 198 (1.0%)
- issuer name: 181 / 198 (91.4%)
- recipient name: 5 / 198 (2.5%)
- total amount: 28 / 198 (14.1%)

#### Method B — Template-routed extraction
Routes documents to coarse invoice templates before applying template-specific extraction rules.

Current coverage:
- invoice number: 18 / 198 (9.1%)
- invoice date: 44 / 198 (22.2%)
- due date: 2 / 198 (1.0%)
- issuer name: 181 / 198 (91.4%)
- recipient name: 17 / 198 (8.6%)
- total amount: 28 / 198 (14.1%)

This improved recipient and invoice-number coverage over the baseline, but gains remain limited.

### Method C — Classical ML candidate ranking
Implemented so far:
- candidate generation for:
  - recipient name
  - invoice number
  - total amount
- labeling batch export
- chunk splitting
- automated API labeling pipeline
- response parser back into labeled CSVs

Current blocker:
- the API-based annotation pipeline is working technically, but OpenAI API quota/billing is unavailable, so candidate labels have not yet been produced at scale
- because of this, the ML rankers are not trained yet


#### Method D — Layout-aware OCR extraction
Uses OCR word coordinates / boxes and zone-based layout heuristics.

Current coverage:
- invoice number: 14 / 198 (7.1%)
- invoice date: 28 / 198 (14.1%)
- due date: 5 / 198 (2.5%)
- issuer name: 51 / 198 (25.8%)
- recipient name: 44 / 198 (22.2%)
- total amount: 56 / 198 (28.3%)

This method shows that layout information helps for recipient and total extraction, but it is still noisy and needs stronger precision controls.

### Main findings so far
- OCR noise and document heterogeneity make invoice extraction much harder than expected
- baseline regex works reasonably for issuer names and some dates, but misses most recipients and many totals
- template routing helps somewhat, especially for recipient and invoice number
- layout-aware extraction increases recall for recipient and total, but precision is still weak
- the most promising next step remains supervised candidate ranking (Method C), but it depends on obtaining labels

### What is left
- clarify with the teacher whether API-assisted candidate annotation is acceptable
- if yes, complete candidate labeling and train ML rankers
- if no, label a smaller subset manually and train on that
- improve Method D precision or keep it as a comparison method
- compare all implemented methods in the report

### Important evaluation note
The current Phase 3 extraction results are measured on the invoice-labeled subset of the processed dataset, not on the full end-to-end classification pipeline. However, manual inspection suggests that this subset still contains OCR-noisy and invoice-adjacent documents, so these metrics should be interpreted as interim extraction coverage on a heterogeneous invoice subset rather than final performance on only clean canonical invoices.

This is especially relevant for:
- `due_date`, which appears to be explicitly present in very few documents
- `issuer_name`, where coverage is high in some methods but correctness still needs manual verification