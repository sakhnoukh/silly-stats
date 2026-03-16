# Dataset Report — Phase 1 Pilot

## Overview
This report documents the pilot dataset used to validate the Phase 1 data collection and preprocessing pipeline.

**Date generated:** 2026-03-16  
**Pipeline status:** All stages passing end-to-end  
**Data source:** Pilot synthetic samples (realistic text templates)

---

## Class Distribution

| Class    | Raw Files | Extracted | Train | Val | Test |
|----------|-----------|-----------|-------|-----|------|
| contract | 20        | 20        | 14    | 3   | 3    |
| email    | 20        | 20        | 14    | 3   | 3    |
| invoice  | 20        | 20        | 14    | 3   | 3    |
| receipt  | 20        | 20        | 14    | 3   | 3    |
| **Total**| **80**    | **80**    | **56**| **12** | **12** |

**Split ratios:** 70% train / 15% val / 15% test (stratified)

---

## Text Statistics (after cleaning)

| Class    | Avg chars | Avg words | Description |
|----------|-----------|-----------|-------------|
| contract | 2384      | 304       | Longest documents — legal language, numbered clauses |
| invoice  | 514       | 86        | Medium length — structured billing fields, line items |
| email    | 425       | 55        | Short — conversational, From/To/Subject headers |
| receipt  | 308       | 56        | Shortest — retail layout, item lists, totals |

---

## Extraction Summary

- **Extraction method:** `direct_text` for all pilot samples (`.txt` files)
- **Failed extractions:** 0
- **Empty documents:** 0
- **Duplicates removed:** 0

---

## TF-IDF Feature Summary

- **Vocabulary size:** 1,369 features
- **Settings:** unigrams + bigrams, max_features=5000, min_df=2, max_df=0.9, sublinear_tf=True
- **Output format:** Sparse CSR matrices (`.npz`)

### Top discriminative features per class

| Class    | Top features |
|----------|-------------|
| contract | agreement, shall, party, parties, services, termination, intellectual |
| email    | subject, please, com, team, meeting, next |
| invoice  | unit price, qty, bill, due, invoice, total due |
| receipt  | x1/x2/x3 (quantities), tel, cashier, shopping, receipt |

---

## Data Sources — Real Dataset Upgrade Path

The pilot uses synthetic text samples. To scale to 150–300 real documents per class:

| Class    | Recommended Source | Access | Action Required |
|----------|--------------------|--------|-----------------|
| invoice  | Kaggle Invoice Dataset / RVL-CDIP | Requires login | Download from Kaggle, place PDFs/images in `data/raw/invoice/` |
| receipt  | SROIE (ICDAR 2019) | GitHub mirror | `git clone https://github.com/zzzDavid/ICDAR-2019-SROIE`, copy images to `data/raw/receipt/` |
| email    | Enron Email Dataset | Public 1.7GB tar.gz | Download from `https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz`, extract ~200 `.txt` files to `data/raw/email/` |
| contract | CUAD | HuggingFace / GitHub | Clone `https://github.com/TheAtticusProject/cuad`, copy PDFs to `data/raw/contract/` |

---

## Known Limitations (Pilot)

1. All pilot documents are `.txt` — PDF and image extraction paths not yet exercised with real data
2. Synthetic samples share structural templates — real data will have more variety
3. TF-IDF vocabulary (1,369) is smaller than expected for real data (target: 3,000–5,000)
4. OCR fallback path not tested (no scanned documents in pilot)

---

## Pipeline Outputs

```
data/
  extracted/dataset_raw.csv       — raw extracted text + metadata
  processed/dataset_clean.csv     — cleaned text + metadata
  processed/dataset.csv           — final labeled dataset
  processed/train.csv             — training split (56 docs)
  processed/val.csv               — validation split (12 docs)
  processed/test.csv              — test split (12 docs)
  features/tfidf_vectorizer.pkl   — fitted TF-IDF vectorizer
  features/X_train.npz            — sparse TF-IDF matrix (train)
  features/X_val.npz              — sparse TF-IDF matrix (val)
  features/X_test.npz             — sparse TF-IDF matrix (test)
  features/y_train.npy            — encoded labels (train)
  features/y_val.npy              — encoded labels (val)
  features/y_test.npy             — encoded labels (test)
  features/label_encoder.pkl      — label encoder
```
