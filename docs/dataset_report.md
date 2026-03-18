# Dataset Report — Phase 1

## Overview

This report summarizes the real dataset assembled for Phase 1.

**Date generated:** 2026-03-18  
**Pipeline status:** All stages passing end-to-end  
**Data source:** Real scanned documents — RVL-CDIP (email, form, invoice) + SROIE (receipt)

---

## Class Distribution

| Class   | Source      | Raw Images | Sampled | Train | Val | Test |
|---------|-------------|-----------|---------|-------|-----|------|
| email   | RVL-CDIP    | 25,020    | 200     | 139   | 30  | 30   |
| form    | RVL-CDIP    | 25,000    | 200     | 139   | 30  | 30   |
| invoice | RVL-CDIP    | 25,020    | 200     | 139   | 29  | 30   |
| receipt | SROIE       | 626       | 200     | 140   | 30  | 30   |
| **Total**| —          | —         | **796** | **557** | **119** | **120** |

**Split ratios:** 70% train / 15% val / 15% test (stratified)

---

## Text Statistics (after cleaning)

| Class   | Avg chars | Avg words | Description |
|---------|-----------|-----------|-------------|
| email   | 588       | 89        | Conversational, From/To/Subject/CC headers, signature blocks |
| form    | 779       | 120       | Structured fields, fax cover sheets, fillable form layouts |
| invoice | 462       | 75        | Billing fields, line items, amounts, due dates |
| receipt | 598       | 102       | Retail layout, item prices, GST/tax totals, Malaysian receipts |

---

## Extraction Summary

| Class   | Method | Failed | Empty dropped |
|---------|--------|--------|---------------|
| email   | Tesseract OCR | 0 | 1 |
| form    | Tesseract OCR | 0 | 4 |
| invoice | Tesseract OCR | 0 | 6 |
| receipt | SROIE box files (pre-extracted) | 0 | 0 |

- OCR ran in parallel (4 workers) on 600 RVL-CDIP images in ~85 seconds
- Receipt text was read directly from annotated bounding-box CSV files — no OCR required
- 11 documents total dropped for empty `clean_text` after cleaning

---

## TF-IDF Feature Summary

- **Vocabulary size:** 5,000 features
- **Settings:** unigrams + bigrams, max_features=5000, min_df=2, max_df=0.9
- **Output format:** Sparse CSR matrices (`.npz`)

### Top discriminative features per class

| Class   | Top features |
|---------|--------------|
| email   | subject, sent, pm, message, 2000, thanks, fw, original |
| form    | date, number, pages, fax, please, cover, information, name, sheet |
| invoice | 00, york, new york, inc, invoice, due, lorillard, street |
| receipt | gst, total, rm, 00, tax, sr, cash, jalan, amount |

---

## Data Sources

| Class   | Dataset | HuggingFace/URL | Label ID | Full size |
|---------|---------|-----------------|----------|-----------|
| email   | RVL-CDIP | `aharley/rvl_cdip` | 2 | 25,020 images |
| form    | RVL-CDIP | `aharley/rvl_cdip` | 1 | 25,000 images |
| invoice | RVL-CDIP | `aharley/rvl_cdip` | 11 | 25,020 images |
| receipt | SROIE ICDAR 2019 | github.com/zzzDavid/ICDAR-2019-SROIE | — | 626 images |

All RVL-CDIP data was streamed from a 38 GB tar.gz archive without downloading the full archive to disk.
Manifests: `data/raw/rvl_cdip_manifest.csv`, `data/raw/rvl_forms_manifest.csv`

---

## Known Limitations

1. 200 samples per class used — full RVL-CDIP has 25K per class if more data is needed
2. OCR quality varies — RVL-CDIP images are historical scanned documents with noise
3. Receipt data is Malaysian retail (SROIE) — may not generalise to other receipt types
4. Invoice class shows most empty extractions (6/200) due to low print quality in some scans

---

## Pipeline Outputs

```
data/
  extracted/dataset_raw.csv       — raw extracted text + metadata
  processed/dataset_clean.csv     — cleaned text + metadata
  processed/dataset.csv           — final labeled dataset
  processed/train.csv             — training split (557 docs)
  processed/val.csv               — validation split (119 docs)
  processed/test.csv              — test split (120 docs)
  features/tfidf_vectorizer.pkl   — fitted TF-IDF vectorizer
  features/X_train.npz            — sparse TF-IDF matrix (train)
  features/X_val.npz              — sparse TF-IDF matrix (val)
  features/X_test.npz             — sparse TF-IDF matrix (test)
  features/y_train.npy            — encoded labels (train)
  features/y_val.npy              — encoded labels (val)
  features/y_test.npy             — encoded labels (test)
  features/label_encoder.pkl      — label encoder
```
