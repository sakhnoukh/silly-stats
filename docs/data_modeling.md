# Phase 2 Report — Modeling & Baseline Evaluation

## Overview

Three baseline classifiers were trained on TF-IDF features (5,000 terms, 1-2 grams) from Phase 1 to classify documents into 4 categories: email, form, invoice, and receipt.

**Best model:** Logistic Regression (macro F1 = 0.893 on test set)

---

## 1. Data Verification

All Phase 1 inputs verified before training:

| Split | Samples | Classes | Empty features | Empty text |
|-------|---------|---------|---------------|------------|
| Train | 557     | 4/4     | 3             | 0          |
| Val   | 119     | 4/4     | 3             | 0          |
| Test  | 120     | 4/4     | 0             | 0          |

Class balance is near-uniform (~139-140 per class in train, 29-30 in val/test).

---

## 2. Model Comparison (Validation Set)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Train Time |
|---|---:|---:|---:|---:|---:|
| MultinomialNB | 0.8235 | 0.8290 | 0.8228 | 0.8206 | 0.01s |
| **LogisticRegression** | **0.8739** | **0.8806** | **0.8762** | **0.8760** | 0.06s |
| LinearSVM | 0.8571 | 0.8611 | 0.8576 | 0.8587 | 0.02s |

All three models perform well above random baseline (0.25 accuracy). Logistic Regression leads on all metrics.

---

## 3. Model Selection

**Selected: Logistic Regression**

Rationale:
- Highest macro F1 on validation set (0.876)
- Best precision and recall balance across all classes
- Highly interpretable (feature coefficients indicate which words drive each class)
- Fast training (0.06s)
- Linear SVM is a close second but slightly lower on all metrics

---

## 4. Final Test Evaluation

| Metric | Score |
|--------|------:|
| Accuracy | 0.8917 |
| Macro Precision | 0.8952 |
| Macro Recall | 0.8917 |
| Macro F1 | 0.8926 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| email | 1.00 | 0.93 | 0.97 | 30 |
| form | 0.79 | 0.77 | 0.78 | 30 |
| invoice | 0.79 | 0.87 | 0.83 | 30 |
| receipt | 1.00 | 1.00 | 1.00 | 30 |

### Confusion Matrix

|  | → email | → form | → invoice | → receipt |
|---|---:|---:|---:|---:|
| **email** | 28 | 2 | 0 | 0 |
| **form** | 0 | 23 | 7 | 0 |
| **invoice** | 0 | 4 | 26 | 0 |
| **receipt** | 0 | 0 | 0 | 30 |

See also: `results/confusion_matrix.png`

---

## 5. Error Analysis

**Total misclassified:** 13/120 (10.8%)

### Easiest and Hardest Classes

- **Easiest:** Receipt (100% accuracy) — distinctive vocabulary: GST, total, RM, cash, tax
- **Second easiest:** Email (93.3%) — strong signals: subject, sent, fw, pm
- **Hardest:** Form (76.7%) — frequently confused with invoice
- **Second hardest:** Invoice (86.7%) — some overlap with form

### Key Findings

1. **Form ↔ Invoice is the main confusion axis** — 7 forms predicted as invoices, 4 invoices predicted as forms. Both are structured documents with similar layouts (numbered fields, addresses, dates). This is the dominant error pattern (11 of 13 errors).

2. **Receipts are never confused** — zero misclassification in either direction. The SROIE receipts have very distinctive vocabulary (GST, RM, Malaysian retail terms) that separates them completely.

3. **Emails are rarely confused** — only 2 misclassified (both predicted as forms). Email-specific vocabulary (subject, sent, fw, original, pm) provides strong discriminative signal.

4. **OCR noise is a contributing factor** — several misclassified documents have garbled text from poor scan quality (e.g., `"pot 3 81440667"`, `"gaa executiy vip yuda"`). Low-quality OCR reduces the useful signal for the classifier.

5. **No invoice-receipt confusion** — despite both being financial documents, they are never mixed up. The vocabulary domains are sufficiently distinct.

### Likely Causes of Form ↔ Invoice Confusion

- Both document types contain structured fields (dates, numbers, addresses)
- Both originate from the same RVL-CDIP tobacco industry archives
- Forms like "purchase orders" and "repair requisitions" share vocabulary with invoices
- Poor OCR on some scans reduces distinguishing text to near-nothing

---

## 6. Summary

The Logistic Regression baseline achieves **89.2% accuracy** and **0.893 macro F1** on the test set, which is strong for a traditional TF-IDF approach on OCR'd scanned documents. The model reliably separates emails and receipts (93-100% accuracy) but struggles with the form/invoice boundary (77-87%).

### Potential Improvements (Optional)

- Add handcrafted features: digit ratio, line count, currency symbol count, keyword presence
- Tune TF-IDF: experiment with higher min_df to reduce OCR noise tokens
- Class-specific keywords: "fax", "cover sheet" for forms; "due", "total due" for invoices
- Increase sample size: RVL-CDIP has 25K images per class — using more could help

---

## Artifacts

```
models/
  multinomialnb.pkl
  logisticregression.pkl
  linearsvm.pkl
  best_model.pkl              (copy of logisticregression.pkl)
  best_model_meta.json

results/
  model_comparison.csv        (3-model validation comparison)
  validation_metrics.txt      (detailed validation reports)
  test_metrics.txt            (final test scores)
  classification_report.txt   (per-class test report)
  confusion_matrix.png        (visual confusion matrix)
  error_analysis.md           (misclassified examples + patterns)
```
