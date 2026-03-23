# Phase 3 — Exploration Findings

## Invoice number
- Labels found in text: e.g. "Invoice No:", "INV-", ...
- Format of the number itself: e.g. numeric only / alphanumeric / with dashes
- Coverage (from script output): X%

## Invoice date
- Labels found: ...
- Date formats seen: e.g. MM/DD/YY, Month DD YYYY
- Coverage: X%

## Due date
- Labels found: ...
- Notes (sometimes just "Net 30" with no explicit date?): ...
- Coverage: X%

## Issuer name
- Where does it appear? (header line? after a label? both?)
- Coverage: X%

## Recipient name  
- Label keywords seen: "Bill To:", "Sold To:", "To:", ...
- Coverage: X%

## Total amount
- Labels seen: "Total", "Amount Due", "Balance", ...
- Currency format: $X,XXX.XX or bare number?
- Coverage: X%

## OCR noise level
- % of docs with significant noise:
- Main noise patterns observed:
```

---

## 🗂️ Where the data actually lives in your repo

Just to be 100% clear on what you have locally:
```
data/
  processed/
    train.csv   ← 557 rows, has "clean_text" + "label" columns  ← USE THIS
    val.csv     ← 119 rows
    test.csv    ← 120 rows
  features/     ← TF-IDF matrices (not needed for extraction)
  raw/          ← manifests only, images were not saved locally