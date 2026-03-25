# Phase 3 — Invoice Information Extraction (Status)

## Goal
Extract the following fields from documents classified as invoices:
- invoice number
- invoice date
- due date
- issuer name
- recipient name
- total amount

---

## Methods implemented

### Method A — Regex / rule-based baseline
- Uses raw OCR text + regex + heuristics
- Strong for issuer names and some dates
- Weak for recipient and total

### Method B — Template-based extraction
- Routes documents into coarse invoice types (lab, law firm, tabular, etc.)
- Improves recipient and invoice number slightly
- Still quite dataset-specific

### Method D — Layout-aware OCR extraction
- Uses OCR boxes (positions) + page zones
- Better for:
  - recipient
  - total amount
- Worse for:
  - issuer
  - invoice metadata

### Method C — Classical ML (candidate ranking)
- Implemented:
  - candidate generation
  - labeling pipeline (automated with Gemini)
- Not trained yet (labels just generated)

---

## Current findings

- OCR noise is very high → makes extraction difficult
- Invoice dataset is **heterogeneous**:
  - not all documents are clean invoices
  - includes invoice-adjacent formats
- Coverage ≠ correctness (important)

Examples:
- issuer_name high coverage → may include wrong values
- due_date very low → often not explicitly present
- layout method improves recall but can be noisy

---

## Important evaluation note

Current results are:
- based on the **invoice-labeled subset**
- not a clean set of canonical invoices
- used for development/debugging

So:
- these are **not final performance metrics**
- we still need a proper evaluation setup

---

## Main issue now

The problem is **evaluation**, not implementation.

We need:
- a clean, fair way to measure correctness
- consistent field definitions
- a small validated subset

---

## Field definitions (important)

- `recipient_name`: billed organization (not issuer, not address, not “remit to”)
- `invoice_number`: true invoice identifier (not reference, not date)
- `due_date`: explicit date only (not "Net 30")
- `payment_terms`: things like Net 30, Due upon receipt
- `issuer_name`: company issuing the invoice
- `total_amount`: final payable amount