# Project Contributors

This document details the contributions of each team member across the five phases of the Document Classification & Invoice Extraction pipeline.

---

## 👥 Team Members & Phase Assignments

### Phase 1: Data Collection & Preprocessing
**Owner:** Sami

**Responsibilities:**
- Download and process large-scale document datasets (RVL-CDIP, SROIE)
- Implement OCR pipeline (Pytesseract) for document text extraction
- Text cleaning: lowercasing, stopword removal, tokenization
- Create train/val/test splits (stratified 70/15/15)
- Generate TF-IDF feature vectors (5K vocabulary, 1-2 grams)

**Deliverables:**
- `scripts/download_rvl_forms.py` — RVL-CDIP downloader
- `scripts/extract_real_data.py` — OCR and text extraction
- `scripts/clean_text.py` — Text preprocessing
- `scripts/build_dataset.py` — Train/val/test splitting
- `scripts/make_features.py` — TF-IDF vectorization
- Processed datasets in `data/processed/` and `data/features/`

**Key Outputs:**
```
- data/processed/train.csv (557 documents, 70%)
- data/processed/val.csv (119 documents, 15%)
- data/processed/test.csv (120 documents, 15%)
- data/features/tfidf_vectorizer.pkl
- data/features/label_encoder.pkl
```

---

### Phase 2: Document Classification Model
**Owner:** Sofia

**Responsibilities:**
- Train classical ML classifiers on TF-IDF features
- Implement and compare multiple model architectures
- Apply dimensionality reduction (PCA, SVD)
- Perform 5-fold cross-validation and hyperparameter tuning
- Generate evaluation metrics (accuracy, F1, confusion matrices)
- Select and save the best-performing model

**Deliverables:**
- `scripts/train_models.py` — Model training pipeline
- `scripts/evaluate_model.py` — Model evaluation
- 10 trained models with multiple variants:
  - Baselines: Logistic Regression, Random Forest, SVM, Naive Bayes
  - PCA variants (150 components)
  - SVD variants (150 components)

**Key Outputs:**
```
- models/best_model.pkl (Logistic Regression baseline)
- models/best_model_meta.json (model metadata)
- results/model_comparison.csv (all 10 models' metrics)
- results/test_metrics.txt (test set performance)
- results/confusion_matrix.png (visualization)
```

**Performance Metrics:**
- **Best Model:** Logistic Regression (Baseline)
- **Test F1-Score:** 1.0 (on synthetic/labeled data)
- **Cross-validation:** 5-fold with consistent performance

---

### Phase 3: Invoice Information Extraction
**Owners:** Bea (beamartin27) & Salmane (salmanemhb)

**Bea's Work (from commits):**
- "phase 3: add exploration script and extraction skeleton" — Initial setup & exploration
- "Add working Phase 3 invoice extraction baseline" — Baseline regex-based extraction
- "Template results" — Template-based extraction method
- "Invoice feature extraction 4 models + results" — Feature extraction & 4 model variants
- "Automated labeling implementation and results" — Candidate labeling pipeline
- Handled diverse invoice formats and layouts

**Salmane's Work (from commits):**
- "Phase 3 complete: gold dataset, ML rankers trained, extraction pipeline, method evaluation" — Built gold dataset for evaluation, trained ML-based rankers, evaluated all extraction methods
- Integrated baseline + template + feature-based methods
- Comprehensive method comparison & evaluation

**Deliverables:**
- `scripts/extract_invoice_fields.py` — Field extraction module
- `scripts/phase_3_extraction/` — Advanced extraction methods
  - `baselines/` — Regex-based baseline
  - `labeling/` — Automated candidate labeling pipeline
  - `ml/` — ML-based candidate ranking
  - `ocr/` — Layout-aware extraction

**6 Extracted Fields:**
1. **Invoice Number** — handles variations (INV-123, inv-123, #123, etc.)
2. **Invoice Date** — parses multiple formats → ISO-8601
3. **Due Date** — payment deadline extraction
4. **Issuer Name** — company/vendor issuing the invoice
5. **Recipient Name** — organization being billed
6. **Total Amount** — final payable amount with currency handling

**Key Outputs:**
```
- results/extraction_coverage.txt (field extraction rates)
- results/invoice_extractions.json (extracted structured data)
- results/invoice_extractions.csv (tabular format)
- results/phase3_extraction/ (method comparison results)
  - baseline/, candidates/, layout/, template/
```

---

### Phase 4: Pipeline Integration & Backend
**Owner:** Matthew

**Responsibilities:**
- Integrate Phase 1-3 into unified end-to-end pipeline
- Build `run.py` as main entry point
- Implement CLI interface with flexible input handling
- Create edge case handling (corrupted files, ambiguous classifications)
- Develop Streamlit web UI for demo (`app.py`)
- Ensure pipeline works with both text and image inputs

**Deliverables:**
- `run.py` — Main unified pipeline script
- `app.py` — Streamlit web interface for interactive demo
- CLI documentation and examples

**Features:**
- ✅ Classify documents into 4 categories
- ✅ Extract invoice fields if classified as "Invoice"
- ✅ Support PDF and image inputs
- ✅ JSON output format
- ✅ Error handling and logging
- ✅ Interactive Streamlit demo

**Usage:**
```bash
# Command line
python3 run.py --input document.pdf --output results.json

# Web UI
streamlit run app.py
```

---

### Phase 5: Documentation & Presentation
**Owner:** Niko

**Responsibilities:**
- Write technical documentation (approach, methodology, results)
- Document end-to-end solution (how to use, architecture)
- Prepare 15-minute presentation with slides
- Create demo script with test cases
- Summarize key findings and performance insights
- Coordinate live demonstration

**Deliverables:**
- `README.md` — Project overview and quick start guide
- `CONTRIBUTORS.md` — This file (team breakdown)
- `docs/data_modeling.md` — Data structure documentation
- `docs/dataset_report.md` — Dataset analysis and statistics
- `docs/exploration_findings.md` — EDA findings
- Presentation slides (15 minutes)
- Live demo walkthrough script

---

## 📊 Summary Statistics

| Phase | Owner | Status | Key Files |
|-------|-------|--------|-----------|
| 1 | Sami | ✅ Complete | `extract_real_data.py`, `clean_text.py`, `make_features.py` |
| 2 | Sofia | ✅ Complete | `train_models.py`, `evaluate_model.py` |
| 3 | Bea & Salmane | ✅ Complete | `extract_invoice_fields.py`, `*_extraction/` |
| 4 | Matthew | ✅ Complete | `run.py`, `app.py` |
| 5 | Niko | ✅ Complete | `docs/`, `README.md`, presentation |

---

## 🤝 Collaboration

### Key Integration Points
1. **Phase 1 → 2:** Processed data from `data/features/` feeds into classifier training
2. **Phase 2 → 4:** Best model (`models/best_model.pkl`) used in unified pipeline
3. **Phase 3 → 4:** Invoice extractor integrated and called when classification predicts "Invoice"
4. **Phase 4 → 5:** `run.py` and `app.py` used for demo and documentation
5. **Throughout:** Continuous integration via Git commits and code review

### Repository Organization
```
Master branch (main)
  ├─ Phase 1 commits (data preparation)
  ├─ Phase 2 commits (classification)
  ├─ Phase 3 commits (extraction)
  ├─ Phase 4 commits (integration)
  └─ Phase 5 commits (documentation + cleanup)
```

---

## 🎓 Learning Outcomes

### Phase 1 (Sami)
- Large-scale data handling and streaming
- OCR technology and text extraction
- Data pipelines and preprocessing workflows
- Feature engineering (TF-IDF)

### Phase 2 (Sofia)
- Classical machine learning algorithms
- Model comparison and hyperparameter tuning
- Cross-validation and performance evaluation
- Dimensionality reduction techniques

### Phase 3 (Bea & Salmane)
**Bea's Learning Outcomes:**
- Regex and pattern matching
- Rule-based systems for structured extraction
- Handling OCR noise and document variation
- Template-based extraction design

**Salmane's Learning Outcomes:**
- Gold dataset creation and validation
- ML model training and evaluation
- Cross-method comparison and analysis
- Pipeline integration and optimization

### Phase 4 (Matthew)
- Software engineering (pipeline integration)
- CLI design and user experience
- Web UI development with Streamlit
- End-to-end system design

### Phase 5 (Niko)
- Technical writing and documentation
- Public speaking and presentation skills
- Project summarization
- Demonstration design

---

## 💾 Git Commits History

Recent commits showing team contributions:

```
45cda01 Added app.py - Simple FrontEnd UI for demo [Matthew]
7e9319d Update run.py - clean demo output formatting [Matthew]
877b3eb Add run.py - Phase 4 pipeline integration [Matthew]
41633e9 Phase 2 and 3 Complete: Classification + Invoice Extraction [Sofia, Bea, Salmane]
3915a0e Phase 3 complete: gold dataset, ML rankers trained, extraction pipeline [Salmane]
729be8b Add working Phase 3 invoice extraction baseline [Bea]
... (and many more from Phases 1-3)
```

Use `git log --author=<name>` to see individual contributions.

---

## 📝 How to Verify Contributions

To check each team member's work:

```bash
# Count commits by author
git log --all --pretty=format:"%an" | sort | uniq -c

# View specific author's commits
git log --author="Sami" --oneline
git log --author="Sofia" --oneline
git log --author="Salmane" --oneline
git log --author="Matthew" --oneline
git log --author="Niko" --oneline

# View changes made by each phase
git log --grep="Phase [1-5]"
```

---

## ✨ Project Achievements

- ✅ From 0 to complete ML pipeline in 5 phases
- ✅ 796 documents processed with high-quality OCR
- ✅ 10+ classifiers trained and compared
- ✅ 6 invoice fields extracted reliably
- ✅ End-to-end working system with web UI
- ✅ Comprehensive documentation
- ✅ No generative AI (pure classical ML)
- ✅ Ready for production use or academic presentation

---

**Last Updated:** April 15, 2026  
**Project:** Document Classification & Invoice Extraction  
**Course:** Statistical Learning (Year 3, Semester 2)
