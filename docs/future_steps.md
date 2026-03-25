# Phase 3 — What to do next

## TL;DR
We already built everything,

Now we need:
1. a small clean evaluation set
2. train the ML ranker
3. evaluate methods fairly

---

## Step 1 — Build a small gold dataset (**MOST IMPORTANT**)

Select ~40–60 invoice documents.

Make sure they include:
- clean invoices
- noisy OCR ones
- tabular invoices
- letter-style invoices
- some ambiguous ones

For each document, manually label:
- invoice number
- invoice date
- due date (or missing)
- issuer name
- recipient name
- total amount

Notes:
- it’s OK if a field is missing → mark as missing
- be consistent with definitions (see phase_3_status.md)

---

## Step 2 — Evaluate existing methods

For those same documents:

Run:
- baseline (Method A)
- template (Method B)
- layout (Method D)

Check:
- is the extracted value correct or not?

Measure:
- per-field accuracy (correct / total)

DO NOT use coverage only.

---

## Step 3 — Train Method C (ML)

Now that we have labels:

Run:
- train_field_rankers.py

Then:
- for each document → pick best candidate per field
- compare with gold values

Measure:
- top-1 accuracy per field

---

## Step 4 — Compare methods

Compare:

| Method | Strengths | Weaknesses |
|--------|----------|-----------|
| Regex  | Simple, stable | misses many fields |
| Template | Better structure | dataset-specific |
| Layout | captures structure | noisy |
| ML (ranker) | most flexible | depends on labels |

---

## Step 5 — Optional improvements (if time)

Only if needed:

- improve layout method:
  - use label → value proximity
  - extract blocks instead of single lines
- try line-level classifier instead of candidate ranking

---

## What NOT to do

- don’t try to perfect regex rules
- don’t chase coverage numbers
- don’t add complex deep learning
- don’t rely on raw dataset as ground truth

---

## Goal for final report

We want to show:
- multiple valid non-generative approaches
- clear comparison
- fair evaluation
- understanding of limitations