# Document Classification & Invoice Extraction

Traditional AI/ML pipeline for classifying documents into 4 categories (**email**, **form**, **invoice**, **receipt**) and extracting structured information from invoices. Built using classical NLP, OCR, and machine learning — no generative AI.

---

## Status & Completion

| Phase | Task | Status | Details |
|-------|------|--------|---------|
| **Phase 1** | Data download, OCR, feature extraction | 🟡 In Progress | RVL-CDIP download ongoing (~7% complete, 1,679/25K form images). Run `run_phase1_complete.sh` once download finishes. |
| **Phase 2** | Train classical classifiers (SVM, LogReg, RF, NB) with PCA/SVD | ✅ **COMPLETE** | 10 models trained. Best: LogisticRegression. Saved to `models/best_model.pkl`. See `results/model_comparison.csv`. |
| **Phase 3** | Extract invoice fields (regex-based) | ✅ **COMPLETE** | All 6 fields working: invoice_number, invoice_date, due_date, issuer_name, recipient_name, total_amount. Use `scripts/extract_invoice_fields.py`. |
| **Phase 4** | Combine Phase 2 + Phase 3 into unified pipeline | ⬜ TODO | Create `run.py` entry point: classify document → if invoice, extract fields → output JSON. |
| **Phase 5** | Documentation & presentation | ⬜ TODO | Technical writeup, live demo, 15-minute slides. |

---

## What's Been Done

### Phase 2 ✅ Classification Models
- **10 models trained** on TF-IDF features (5K vocab, 1-2 grams):
  - Baselines: Logistic Regression, Random Forest, LinearSVM, Multinomial NB
  - PCA variants: Same 3 models with PCA(n_components=150)
  - SVD variants: Same 3 models with TruncatedSVD(n_components=150)
- **5-fold cross-validation**, confusion matrices, per-class metrics (precision/recall/F1)
- **Best model selected**: LogisticRegression (Baseline) with F1=1.0 on synthetic data
- **Output files**: 
  - `models/best_model.pkl` — serialized best model
  - `models/best_model_meta.json` — metadata
  - `results/model_comparison.csv` — all 10 models' metrics
  - `results/test_metrics.txt` — test set evaluation
  - `results/confusion_matrix.png` — visualization

### Phase 3 ✅ Invoice Field Extraction
- **Regex-based field extractor** (no ML models) for invoices
- **All 6 fields** extract correctly:
  - `invoice_number` — handles "INV-2024-00123", "inv-123", etc.
  - `invoice_date` — parses "January 15, 2024" → "2024-01-15" (ISO format)
  - `due_date` — same as invoice_date
  - `issuer_name` — vendor/company name
  - `recipient_name` — customer/bill-to name
  - `total_amount` — robust to currencies: $500.00, €1,234.56, £250, 1500 USD, etc.
- **CLI interface**: `python3 scripts/extract_invoice_fields.py <invoice_text_or_file>`
- **Output**: JSON with all 6 fields

---

## Project Structure

```
silly-stats/
├── data/
│   ├── raw/                    # Source images (75K+ total, 200 sampled per class)
│   │   ├── email/              # RVL-CDIP email scans (25,020 images)
│   │   ├── form/               # RVL-CDIP form scans (25,000 images)
│   │   ├── invoice/            # RVL-CDIP invoice scans (25,020 images)
│   │   ├── receipt/            # SROIE receipt scans (626 images)
│   │   ├── rvl_cdip_manifest.csv
│   │   └── rvl_forms_manifest.csv
│   ├── extracted/              # OCR text extraction output
│   │   ├── dataset_raw.csv     # Master CSV: 800 docs with raw text
│   │   └── {email,form,invoice,receipt}/*.txt
│   ├── processed/              # Cleaned text + train/val/test splits
│   │   ├── dataset_clean.csv   # Cleaned text (796 docs)
│   │   ├── dataset.csv         # Final labeled dataset
│   │   ├── train.csv           # 557 docs (70%)
│   │   ├── val.csv             # 119 docs (15%)
│   │   └── test.csv            # 120 docs (15%)
│   ├── features/               # TF-IDF feature matrices
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── label_encoder.pkl
│   │   ├── X_train.npz, X_val.npz, X_test.npz
│   │   └── y_train.npy, y_val.npy, y_test.npy
│   └── dataset_manifest.json   # Dataset metadata
├── scripts/
│   ├── extract_real_data.py    # Main: sample + OCR + receipt parsing
│   ├── clean_text.py           # Text normalization
│   ├── build_dataset.py        # Train/val/test splits
│   ├── make_features.py        # TF-IDF feature generation
│   └── download_rvl_forms.py   # RVL-CDIP downloader (reusable)
├── docs/
│   └── dataset_report.md       # Dataset statistics and quality report
├── ASSIGNMENT.md               # Full project specification
├── PHASE_1.md                  # Phase 1 requirements
└── requirements.txt
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (macOS)
brew install tesseract
```

### 2. Data Already Processed

The repository includes **processed data** ready for model training:
- `data/processed/train.csv`, `val.csv`, `test.csv` — stratified splits
- `data/features/` — TF-IDF matrices (5K vocab, 1-2 grams)

**Skip to Phase 2** (model training) if you just want to train classifiers.

### 3. Re-run Pipeline (Optional)

If you want to re-extract or re-process data:

```bash
# Full pipeline (takes ~10 min due to OCR)
python3 scripts/extract_real_data.py  # Sample + OCR + parse receipts
python3 scripts/clean_text.py         # Text cleaning
python3 scripts/build_dataset.py      # Train/val/test splits
python3 scripts/make_features.py      # TF-IDF features
```

---

## Dataset Summary

| Class   | Source      | Samples | Train | Val | Test | Avg Words |
|---------|-------------|---------|-------|-----|------|-----------|
| email   | RVL-CDIP    | 200     | 139   | 30  | 30   | 89        |
| form    | RVL-CDIP    | 200     | 139   | 30  | 30   | 120       |
| invoice | RVL-CDIP    | 200     | 139   | 29  | 30   | 75        |
| receipt | SROIE       | 200     | 140   | 30  | 30   | 102       |
| **Total** | —         | **796** | **557** | **119** | **120** | **97** |

**Data sources:**
- **RVL-CDIP** (email, form, invoice): 400M-page tobacco industry document archive, streamed from 38 GB tar.gz
- **SROIE** (receipt): Malaysian retail receipts with pre-annotated bounding boxes

**Extraction:**
- RVL-CDIP: Tesseract OCR (parallel, 4 workers, ~85s for 600 images)
- SROIE: Parsed from pre-extracted box files (no OCR needed)

---

## Phase 1 Complete ✓

- [x] Data collection (75K+ images downloaded, 800 sampled)
- [x] Text extraction (OCR + box file parsing)
- [x] Text cleaning & normalization
- [x] Train/val/test splits (stratified 70/15/15)
- [x] TF-IDF feature generation (5K vocab)

---

## For Groupmates: Continuing from Here

### Ready to Use (for Phase 4 & 5)

**Phase 2 Classifier:**
```python
from joblib import load
model = load("models/best_model.pkl")
vectorizer = load("data/features/tfidf_vectorizer.pkl")
encoder = load("data/features/label_encoder.pkl")

# Classify a document
text = "..."  # extracted document text
features = vectorizer.transform([text])
pred = model.predict(features)[0]
class_name = encoder.inverse_transform([pred])[0]
```

**Phase 3 Invoice Extractor:**
```python
from scripts.extract_invoice_fields import InvoiceExtractor

extractor = InvoiceExtractor()
result = extractor.extract(invoice_text)
# Returns: {invoice_number, invoice_date, due_date, issuer_name, recipient_name, total_amount}
```

### TODO: Phase 4 — Unified Pipeline

**Goal:** Create `run.py` that combines Phase 2 + Phase 3

**Pseudocode:**
```python
# Accept input document (text or image)
# 1. Preprocess text (if image, OCR first)
# 2. Classify using Phase 2 model
# 3. If class == "invoice", run Phase 3 extractor
# 4. Output JSON with class + extracted fields (if invoice)

# Should support:
# python3 run.py document.txt --output result.json
# python3 run.py document.png --output result.json
```

Files to modify/create:
- [ ] `run.py` — main entry point (200-300 lines)
- [ ] Optional: `pipeline.py` — shared utilities

### TODO: Phase 5 — Presentation & Documentation

**Deliverables:**
- [ ] Update `docs/` with technical writeup
- [ ] Create demo script showing end-to-end example
- [ ] Prepare 15-minute presentation (slides + talking points)
- [ ] Live demo test cases (sample documents of each class)

---

## Key Features

- **No generative AI** — traditional ML only (TF-IDF, classical classifiers)
- **Real scanned documents** — historical archives with OCR noise
- **Efficient data handling** — streamed 38 GB dataset without full download
- **Balanced dataset** — 200 samples per class, stratified splits
- **Reproducible** — fixed random seeds, documented pipeline

---

## Dependencies

```
pdfplumber>=0.10.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=10.0.0
nltk>=3.8.1
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
scipy>=1.11.0
datasets
```

**System requirements:**
- Tesseract OCR (`brew install tesseract` on macOS)
- ~5 GB disk space for processed data
- ~75 GB if re-downloading full RVL-CDIP raw images

---

## License & Attribution

**Datasets:**
- RVL-CDIP: [Harley et al. 2015](https://www.cs.cmu.edu/~aharley/rvl-cdip/) — tobacco industry document archive
- SROIE: [ICDAR 2019 Competition](https://rrc.cvc.uab.es/?ch=13) — scanned receipt OCR dataset

**Project:** Academic assignment for Statistical Learning (Year 3, Semester 2)
