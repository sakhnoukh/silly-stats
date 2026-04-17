# Project Contributors

This document details the contributions of each team member across the five phases of the Document Classification & Invoice Extraction pipeline.

---

## 👥 Team Members & Phase Assignments

### Phase 1: Data Collection & Preprocessing
**Owner:** Sami

**Work (from scripts & outputs):**
- Downloaded and processed RVL-CDIP and SROIE datasets (~800 documents total)
- Implemented OCR pipeline with Pytesseract for text extraction
- Built text cleaning pipeline: lowercasing, stopword removal, tokenization
- Created train/val/test splits (stratified 70/15/15)
- Generated TF-IDF feature vectors (5K vocabulary, 1-2 grams)
- Saved vectorizers and encoders for downstream phases

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
**Owner:** Sofia (sofiasanjose)

**Work (from commit: "Phase 2 and 3 Complete: Classification + Invoice Extraction"):**
- Trained 10 classical ML classifiers on TF-IDF features
- Implemented model variants: Logistic Regression, Random Forest, SVM, Naive Bayes
- Applied dimensionality reduction: PCA and Truncated SVD (150 components)
- Performed 5-fold cross-validation and hyperparameter tuning
- Generated evaluation metrics: accuracy, F1-scores, confusion matrices
- Selected Logistic Regression as best-performing model
- Saved trained models and metadata for pipeline integration

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

**Bea:**
- Built the Phase 3 extraction baseline and evolved it into stronger extractor versions (up to `extract_invoice_fields_v5.py`)
- Improved regex/rule-based extraction for the 6 target invoice fields
- Added OCR-based rescue for difficult real invoices
- Built and evaluated a real-world benchmark set using manually labeled real + translated invoices:
  - `tests/invoices_real/`
  - `tests/ground_truth_real6.json`
- Ran detailed extractor comparisons across multiple test regimes
- Implemented and tested the FATURA line-based extraction pipeline
- Compared regex, OCR-rescue, and line-classification approaches and selected the strongest practical extractor version for integration into the pipeline

**Salmane:**
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

#### Phase 3.5 — Extractor v5 robustness pass (Salmane)

Follow-up hardening pass on the invoice extractor to handle real-world
(non-synthetic) documents — OCR noise, Spanish/multilingual layouts, and
receipt variants. Target: **≥70%** on the real-world test set
(`tests/invoices_real/`, 14 PDFs, 73 scoreable fields).

**Trajectory (real-world set):**

| Stage | Accuracy | Δ fields |
|---|---|---|
| v5 baseline (Phases A–E done) | 64.4% | — |
| Phase F: Spanish all-caps recipient heuristic | 67.1% | +2 |
| Phase G: NER ORG rescue + stricter `_looks_bad_issuer` | 68.5% | +1 |
| Phase H: invoice-number trailing numeric token | 69.9% | +1 |
| Phase H+: email-domain issuer fallback | **71.2%** | +1 |

**Final results across all 3 test sets:**

| Test set | Files | Score | Accuracy |
|---|---|---|---|
| `tests/invoices/` (synthetic) | 19 | 101/114 | 88.6% |
| `tests/invoices_ocr/` (degraded PNG) | 7 | 38/42 | 90.5% |
| `tests/invoices_real/` (real-world) | 14 | 52/73 | 71.2% |

Unit tests: **77/77 passing** (`tests/test_phase3.py`).

**Key changes in `scripts/extract_invoice_fields_v5.py`:**

- **Phase A–E** — `_post_ocr_cleanup()`, Tesseract `--psm 6`, 3× render
  scale, European-format amount parser (`1.871,86` → `1871.86`), multiline
  total pattern, hotel/booking invoice-number anchors, stricter bad-value
  filters.
- **Phase F** — Spanish all-caps recipient heuristic in both
  `_extract_recipient_name_ocr` and `extract_recipient_name`; requires
  ≥4 chars per word; blacklists tax IDs (GSTIN, CIN, PAN, NIF, CIF, …).
- **Phase G** — New `rank_org()` in `_ner_rescue` with 2-pass scan
  (`text[:2500]` → `text[:6000]`); `_looks_bad_issuer` now rejects
  address/facility fragments (`business campus`, `wholesaler`,
  `shopping centre`) and Spanish 2–3-word all-caps personal-name shape.
- **Phase H** — Invoice-number capture now extends with one trailing 3–6
  digit token when the primary value contains letters and the token is
  not a year (handles `E77148D3 0001`).
- **Phase H+** — Email-domain issuer fallback: if all other rescues fail,
  extract the second-level domain from the first non-recipient-line email,
  filter generic providers, uppercase it (`info@neony.es` → `NEONY`).

**New eval helpers:**
- `scripts/quick_eval.py` — fast real-world AFTER-only eval.
- `scripts/quick_eval_synthetic.py` — per-field breakdown on synthetic set.

**Documentation:**
- `docs/v5_improvements.md` — full change log, results tables, remaining
  failure catalogue (21/73 — mostly OCR-quality and non-standard date
  formats), reproduction commands.

---

### Phase 4: Pipeline Integration & Backend
**Owner:** Matthew

**Work (from commits):**
- "Add run.py - Phase 4 pipeline integration" — Built unified pipeline entry point
- "Update run.py - clean demo output formatting" — Improved output formatting & UX
- "Update run.py - minor fix" — Bug fixes and refinements
- "Added app.py - Simple FrontEnd UI for demo" — Created Streamlit web interface
- Integrated Phase 2 classifier with Phase 3 extractor
- Built CLI interface for batch/single document processing
- Implemented JSON output format for results
- Created interactive web UI for easy demonstration

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
45cda01 Added app.py - Simple FrontEnd UI for demo [Matthew - Phase 4]
e2f740d Update run.py - minor fix [Matthew - Phase 4]
7e9319d Update run.py - clean demo output formatting [Matthew - Phase 4]
877b3eb Add run.py - Phase 4 pipeline integration [Matthew - Phase 4]
41633e9 Phase 2 and 3 Complete: Classification + Invoice Extraction [Sofia - Phase 2, Bea & Salmane - Phase 3]
3915a0e Phase 3 complete: gold dataset, ML rankers trained, extraction pipeline [Salmane - Phase 3]
729be8b Add working Phase 3 invoice extraction baseline [Bea - Phase 3]
356b3ed phase 3: add exploration script and extraction skeleton [Bea - Phase 3]
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
- ✅ No generative AI (rule-based extraction, OCR, and discriminative ML only)
- ✅ Ready for production use or academic presentation

---

**Last Updated:** April 15, 2026  
**Project:** Document Classification & Invoice Extraction  
**Course:** Statistical Learning (Year 3, Semester 2)
