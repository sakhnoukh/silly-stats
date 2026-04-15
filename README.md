# Document Classification & Invoice Extraction System

A complete machine learning pipeline that classifies documents into 4 categories (**Invoice**, **Receipt**, **Email**, **Contract**) and extracts structured fields from invoices using classical NLP, OCR, and machine learning — **no generative AI**.

---

## 📋 Project Overview

### Goal
Build an end-to-end system that:
1. Classifies documents into 4 types
2. Extracts 6 key fields **only from invoices**: invoice number, invoice date, due date, issuer name, recipient name, total amount

### Constraints
- Classical ML only (SVM, Random Forest, Logistic Regression)
- NLP-based feature extraction (TF-IDF, regex, tokenization)
- No generative AI or large language models
- Handle diverse document formats and OCR noise

---

## ✅ Project Status

| Phase | Description | Owner | Status |
|-------|-------------|----------|--------|
| **Phase 1** | Data Collection & Preprocessing | Sami | ✅ **COMPLETE** |
| **Phase 2** | Classification Models | Sofia | ✅ **COMPLETE** |
| **Phase 3** | Invoice Field Extraction | Salmane | ✅ **COMPLETE** |
| **Phase 4** | Pipeline Integration & Demo | Matthew | ✅ **COMPLETE** |
| **Phase 5** | Documentation & Presentation | Niko | ✅ **COMPLETE** |

---

## 🏗️ Architecture

```
Input Document (PDF/Image)
  ↓
[Phase 1] OCR & Text Extraction
  ↓
[Phase 2] Classification (TF-IDF + Model)
  ↓
  └─→ If Invoice: [Phase 3] Extract Fields (Regex + Rules)
  └─→ Else: Return classification only
  ↓
[Phase 4] Output Results (JSON)
```

---

## 🚀 Quick Start

### Requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Complete Pipeline
```bash
# Start the Streamlit demo app
streamlit run app.py

# Or run classification + extraction directly
python3 run.py --input sample.pdf
```

### Run Individual Phases
```bash
# Phase 2: Classify a document
python3 scripts/evaluate_model.py data/sample_invoice.txt

# Phase 3: Extract invoice fields
python3 scripts/extract_invoice_fields.py data/sample_invoice.txt
```

---

## 📊 Results Summary

### Phase 2: Classification Performance
**Best Model:** Logistic Regression (Baseline)
- 10 models trained with 5-fold cross-validation
- PCA and SVD dimensionality reduction tested
- Confusion matrices and per-class metrics in `results/model_comparison.csv`

### Phase 3: Invoice Extraction
**6 fields extracted** with high accuracy:
- Invoice Number, Invoice Date, Due Date
- Issuer Name, Recipient Name, Total Amount
- Regex-based with rule-driven parsing
- Handles currency variations ($, €, £, USD, etc.)

---

## 🗂️ Project Structure

```
silly-stats/
├── README.md                          # Project overview
├── CONTRIBUTORS.md                    # Team breakdown by phase
├── requirements.txt                   # Python dependencies
│
├── data/                              # Dataset storage
│   ├── raw/                           # Original documents
│   ├── processed/                     # Cleaned & split data
│   └── features/                      # TF-IDF vectorizers (.pkl)
│
├── scripts/                           # Main pipeline code
│   ├── extract_real_data.py          # [Phase 1] OCR & text extraction
│   ├── clean_text.py                 # [Phase 1] Text preprocessing
│   ├── build_dataset.py              # [Phase 1] Create train/val/test splits
│   ├── make_features.py              # [Phase 1] Generate TF-IDF features
│   ├── train_models.py               # [Phase 2] Train classifiers
│   ├── evaluate_model.py             # [Phase 2] Model evaluation
│   ├── extract_invoice_fields.py     # [Phase 3] Field extraction
│   ├── run.py                        # [Phase 4] Main pipeline entry point
│   └── app.py                        # [Phase 4] Streamlit demo UI
│
├── models/                            # Trained models
│   ├── best_model.pkl                # Best classifier
│   └── best_model_meta.json          # Model metadata
│
├── results/                           # Pipeline outputs
│   ├── model_comparison.csv          # Classification metrics
│   ├── test_metrics.txt              # Test set results
│   └── phase3_extraction/            # Extraction results by method
│
└── docs/                              # Technical documentation
    ├── data_modeling.md              # Data structure docs
    ├── dataset_report.md             # Dataset analysis
    └── exploration_findings.md       # EDA findings
```

---

## 🔬 Technical Approach

### Phase 1: Data Preprocessing
- OCR extraction using Pytesseract/PyPDF2
- Text cleaning: lowercasing, stopword removal, tokenization
- TF-IDF vectorization (5K vocabulary, 1-2 grams)
- Train/val/test splits (70/15/15)

### Phase 2: Classification
- **Models Tested:** Logistic Regression, SVM, Random Forest, Naive Bayes
- **Dimensionality Reduction:** PCA and Truncated SVD (150 components)
- **Evaluation:** 5-fold cross-validation, confusion matrices, F1-scores

### Phase 3: Information Extraction
- Regex patterns for structured fields
- Rule-based parsing for dates and amounts
- Handles layout variations (letter-style, tabular, etc.)
- Robust to OCR noise and formatting inconsistencies

### Phase 4: Integration
- Unified `run.py` entry point
- CLI interface for batch/single processing
- JSON output format
- Streamlit web UI for easy demo

---

## 📈 Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point for pipeline |
| `app.py` | Streamlit demo web interface |
| `models/best_model.pkl` | Trained classification model |
| `data/features/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `results/model_comparison.csv` | Model performance metrics |

---

## 🔧 Configuration

Edit `run.py` or use environment variables:
```python
INPUT_PDF = "path/to/document.pdf"
OUTPUT_JSON = "path/to/results.json"
VERBOSITY = "debug"  # or "info"
```

---

## 📝 Notes

- Phase 1 downloads from RVL-CDIP dataset (~25K form images)
- Classification trained on synthetic + real documents
- Extraction methods compared: Baseline (regex), Template, Layout-aware, ML-based
- All models are deterministic (no randomness after seeding)

---

## 👥 Team & Contributions

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed phase assignments and individual team member contributions.

---

**Last Updated:** April 15, 2026
