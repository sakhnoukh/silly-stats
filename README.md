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
| **Phase 3** | Invoice Field Extraction + Labeling | Bea & Salmane | ✅ **COMPLETE** |
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

## � System Requirements & Installation

### System Requirements
- **OS:** macOS, Linux, or Windows with WSL
- **Python:** 3.8 or higher
- **Memory:** 4 GB RAM minimum (8 GB recommended)
- **Disk Space:** 2 GB for processed data + models

### System Dependencies (macOS)
```bash
# Install Tesseract OCR
brew install tesseract

# Verify installation
tesseract --version
```

### System Dependencies (Linux - Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### System Dependencies (Windows)
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

---

## 📦 Python Dependencies

All required packages are listed in `requirements.txt`:

```
pdfplumber>=0.10.0          # PDF text extraction
pytesseract>=0.3.10         # OCR (wrapper for Tesseract)
pdf2image>=1.16.3           # PDF to image conversion
Pillow>=10.0.0              # Image processing
nltk>=3.8.1                 # Natural Language Toolkit
scikit-learn>=1.3.0         # Classical ML models
pandas>=2.0.0               # Data manipulation
numpy>=1.24.0               # Numerical computing
joblib>=1.3.0               # Model serialization
scipy>=1.11.0               # Scientific computing
streamlit>=1.28.0           # Web UI framework
datasets                    # HuggingFace datasets library
```

---

## 🚀 Installation & Setup (Step-by-Step)

### Step 1: Clone or Navigate to Repository
```bash
cd /path/to/silly-stats
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # On macOS/Linux
# OR
venv\Scripts\activate             # On Windows
```

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Test Python packages
python3 -c "import sklearn, pandas, streamlit; print('✓ All packages installed')"

# Test Tesseract OCR
tesseract --version
```

---

## ▶️ How to Run the Complete System

### Option 1: Interactive Web Demo (Recommended for Teachers)

**Easiest way to demo the entire system to the class:**

```bash
# Make sure venv is activated
source venv/bin/activate

# Start the Streamlit web UI
streamlit run app.py
```

**What you'll see:**
- Web interface opens at `http://localhost:8501`
- Upload a document (PDF or image)
- System classifies it into: Invoice, Receipt, Email, or Contract
- If it's an invoice, structured fields are extracted automatically
- Results displayed in JSON format

### Option 2: Command-Line Pipeline

**For batch processing or scripting:**

```bash
# Activate virtual environment
source venv/bin/activate

# Run the unified pipeline on a single document
python3 run.py --input path/to/document.pdf --output results.json

# Example:
python3 run.py --input sample_invoice.pdf --output extracted_data.json
```

**Output:**
```json
{
  "classification": "Invoice",
  "confidence": 0.98,
  "extracted_fields": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "issuer_name": "Company ABC Inc.",
    "recipient_name": "Customer XYZ Ltd.",
    "total_amount": "$1,500.00"
  }
}
```

### Option 3: Run Individual Pipeline Phases

**For debugging or understanding each phase:**

```bash
source venv/bin/activate

# Phase 2: Classify a document
python3 scripts/evaluate_model.py path/to/document.txt

# Phase 3: Extract invoice fields from text
python3 scripts/extract_invoice_fields.py path/to/invoice.txt

# Phase 1: Regenerate features (if needed)
# python3 scripts/make_features.py
```

---

## 🧪 Quick Test (Demo with Sample Data)

### Test 1: Classification Only
```bash
source venv/bin/activate

# Create a sample text file
echo "Invoice #INV-123456 dated January 15, 2024. Total: $5,000.00" > sample.txt

# Classify it
python3 scripts/evaluate_model.py sample.txt
```

**Expected output:** Classification as "Invoice" with high confidence

### Test 2: Full Pipeline
```bash
# Run the complete pipeline on the sample
python3 run.py --input sample.txt --output sample_output.json

# View the results
cat sample_output.json
```

**Expected output:** Classified as Invoice + extracted fields in JSON

### Test 3: Web Demo
```bash
# Start the interactive web UI
streamlit run app.py

# Then:
# 1. Open http://localhost:8501 in browser
# 2. Upload or paste a document
# 3. See live classification + extraction results
```

---

## 🔍 Verifying the Setup

Run this script to verify everything is working:

```bash
source venv/bin/activate

python3 << 'EOF'
import sys
print("Python version:", sys.version)

# Check packages
packages = ['sklearn', 'pandas', 'streamlit', 'pytesseract', 'nltk']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} installed")
    except ImportError:
        print(f"✗ {pkg} NOT installed")

# Check models exist
from pathlib import Path
files = [
    'models/best_model.pkl',
    'data/features/tfidf_vectorizer.pkl',
    'scripts/run.py',
    'scripts/app.py'
]
for f in files:
    exists = Path(f).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {f}")
    
print("\n✓ Setup verification complete!")
EOF
```

---

## 📊 Expected Outputs

### Classification Results
```
Document: invoice.pdf
Classification: Invoice
Confidence: 0.95
Processing Time: 0.32s
```

### Extraction Results (Invoice)
```json
{
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15",
  "due_date": "2024-02-15",
  "issuer_name": "ABC Corporation",
  "recipient_name": "XYZ Limited",
  "total_amount": "$5,000.00"
}
```

---

## 🛠️ Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'X'`
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Tesseract not found
**Solution:** Install Tesseract separately:
```bash
# macOS
brew install tesseract

# Linux (Ubuntu)
sudo apt-get install tesseract-ocr

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: Streamlit says "port 8501 already in use"
**Solution:** Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: "best_model.pkl not found"
**Solution:** Make sure you're running from the project root directory:
```bash
pwd  # Should show /path/to/silly-stats
ls models/best_model.pkl  # Should exist
```

---

## �👥 Team & Contributions

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed phase assignments and individual team member contributions.

---

**Last Updated:** April 15, 2026
