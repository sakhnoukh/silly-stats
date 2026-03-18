# Document Classification & Invoice Extraction

Traditional AI/ML pipeline for classifying documents into 4 categories (**email**, **form**, **invoice**, **receipt**) and extracting structured information from invoices. Built using classical NLP, OCR, and machine learning — no generative AI.

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

**Next:** Phase 2 — Train classification models (SVM, Logistic Regression, Random Forest)

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
