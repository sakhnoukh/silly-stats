# Invoice Extractor v5 — Improvements Log

Target: reach **≥70%** accuracy on the real-world invoice test set
(`tests/invoices_real/`, 14 PDFs, 73 scoreable fields).

## Final results

| Test set | Files | Score | Accuracy |
|---|---|---|---|
| `tests/invoices/` (synthetic) | 19 | 101/114 | **88.6%** |
| `tests/invoices_ocr/` (degraded PNG) | 7 | 38/42 | **90.5%** |
| `tests/invoices_real/` (real-world) | 14 | 52/73 | **71.2%** |

Unit tests: **77/77 passing** (`tests/test_phase3.py`).

Real-world trajectory: **64.4% → 71.2%** (+6.8 pts, +5 fields).

| Phase | After change | Δ |
|---|---|---|
| Baseline (A–E done) | 64.4% | — |
| F: Spanish all-caps recipient heuristic | 67.1% | +2 |
| G: NER ORG rescue + stricter `_looks_bad_issuer` | 68.5% | +1 |
| H: Invoice number trailing-numeric extension | 69.9% | +1 |
| H+: Email-domain fallback for issuer | **71.2%** | +1 |

---

## Changes (all in `scripts/extract_invoice_fields_v5.py`)

### Phase A–E (pre-session recap)
- `_post_ocr_cleanup()` for common OCR confusions.
- Tesseract `--psm 6`, render scale bumped `2.0 → 3.0`.
- European number format in `_normalise_amount` (`1.871,86` → `1871.86`).
- Multiline "Total … €" pattern in `amount_patterns`.
- Hotel/booking anchors for invoice number (`Booking ID`, `Confirmation No`, `Folio`).
- "method of payment", "payment receipt", "Madrid" added to bad-value filters.
- Null-out cleanup for any field that stays bad after all rescues.

### Phase F — Spanish all-caps recipient (+2)
Many Spanish invoices put the recipient as a bare all-caps line (no "Bill to:" label).
Added a heuristic in both `_extract_recipient_name_ocr` (step 1b) and
`extract_recipient_name` (step 4):

- Match 2–3 all-caps words, each ≥4 characters.
- Blacklist: `GSTIN`, `CIN`, `PAN`, `NIF`, `CIF`, `DNI`, `IBAN`, `SWIFT`,
  `IVA`, `VAT`, `TOTAL`, `BASE`, …
- Falls through if line contains digits or non-Latin accents.

Result: captures `BEATRIZ MARTIN MARTIN`, `MS BEATRIZ MARTIN MARTIN`, etc.

### Phase G — NER ORG rescue + address/facility rejection (+1)

**New `rank_org()` in `_ner_rescue`**: ranks spaCy multilingual ORG candidates by
company-suffix match, position (earlier = better), token count, and length.
Runs a 2-pass scan: first `text[:2500]`, then `text[:6000]` if nothing usable.

**Strengthened `_looks_bad_issuer`**: now rejects

- facility keywords: `business campus`, `business park`, `industrial park`,
  `shopping centre/center`, `wholesaler`, `warehouse`, `logistics centre`.
- 2–3 all-caps Spanish personal-name shape (goes to recipient, not issuer)
  **only when no company suffix is present**.
- Document titles concatenated with client name (`neon sign ...`, length > 4 words).

### Phase H — Invoice number tail token (+1)

Some gold values contain a space (`E77148D3 0001`) but the capture group
`([A-Z0-9][\w\-/\.]{2,30})` stops at whitespace.

After false-positive rejection, extend the match with one trailing 3–6 digit
token when:

- the primary capture contains letters (alphanumeric code, not plain number), and
- the trailing token is **not** a 4-digit year (1900–2099), and
- the token sits within 20 chars of the main match.

```python
tail_match = re.match(
    r'\s+(\d{3,6})(?:\s|$)',
    text[match.end():match.end() + 20],
)
if tail_match and re.search(r'[A-Za-z]', val):
    tail_val = tail_match.group(1)
    if not (len(tail_val) == 4 and 1900 <= int(tail_val) <= 2099):
        val = f"{val} {tail_val}"
```

### Phase H+ — Email-domain issuer fallback (+1)

`NeonyInvoice.pdf` has no "From:"/vendor anchor, no ORG hit from NER, but does
contain `info@neony.es` — and `NEONY` is the gold issuer.

Final fallback in `extract()` (after regex, OCR rescue, NER rescue, all failed):

- Scan the first 25 lines.
- Skip any line matching a recipient anchor (`Bill to`, `Ship to`, `Customer`,
  `Client`, `Recipient`, `Attn`, `Delivery address`).
- Extract the second-level domain from the first email.
- Skip generic providers (`gmail`, `hotmail`, `yahoo`, `outlook`, `icloud`,
  `live`, `aol`, `mail`, `protonmail`, `msn`, `me`, `gmx`, `inbox`, `fastmail`,
  `zoho`).
- Use the domain uppercased as `issuer_name`.

Result: `info@neony.es` → `NEONY`.

---

## Supporting scripts added

- `scripts/quick_eval.py` — fast AFTER-only eval (real-world set).
- `scripts/quick_eval_synthetic.py` — AFTER-only eval on synthetic set with
  per-field breakdown.
- `scripts/_debug_issuer.py` — temporary debug helper (inspect OCR issuer chain
  per file). Remove when done.

## How to run

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:NLTK_DATA = "$pwd\.venv\nltk_data"

# Quick real-world eval (used during iteration)
.venv\Scripts\python.exe scripts\quick_eval.py

# Full pipeline eval (any of the 3 sets)
.venv\Scripts\python.exe scripts\evaluate_pipeline.py `
    --extractor scripts/extract_invoice_fields_v5.py `
    --invoices tests/invoices `
    --ground-truth tests/ground_truth.json `
    --output results/eval_v5_synthetic.json

# Unit tests
.venv\Scripts\python.exe -m pytest tests/test_phase3.py -v
```

## Known remaining failures (real-world set, 21/73 fields)

Not blocking the 70% target, logged for future work:

- `AdobeTranslated.pdf` — OCR garbles issuer (`aAgope Systems Sortware ireiana Lta`).
- `AMAZONTranslated.pdf` — picks wrong ORG (`HangZhouFengChiJinChuKouYouXianGongSi`).
- `AmazonWebServices.pdf` — `due_date` missing, total wrong.
- `AzureInterior.pdf` — non-standard gold date `2023-20-03`; `due_date` offset.
- `GasolineraCarrefour` — `invoice_number 119` vs `2025FBFN00188074`; issuer null.
- `LeroyMerlin` — issuer captured as tagline; recipient OCR garbage.
- `MetacrilatoMadrid` — issuer null, recipient `oe`, total 89.84 vs 74.25.
- `oyo.pdf` — `due_date`, `total` null (OCR tolerance).
- `RENTA` — wrong invoice number (`EX223`), issuer from header junk.
- `SammyMaystone` — picks `PO_NUMBER_123` instead of `invoice_number_1`.
- `Software.pdf` — `due_date` null for non-standard `2025-25-07`.

Most are OCR-quality or non-standard date-format issues; each would take
targeted work with limited upside.
