"""
Phase 3 — Invoice Data Exploration
Run this BEFORE writing extract_invoices.py.
Goal: understand what the OCR text actually looks like for invoice documents.

Usage:
    python3 scripts/explore_invoices.py
"""

import os
import re
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(PROJ_ROOT, "data", "processed")

# ── 1. Load all splits and isolate invoices ────────────────────────────────

train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
invoices = all_df[all_df["label"] == "invoice"].copy()
invoices["clean_text"] = invoices["clean_text"].fillna("")

print(f"Total invoice documents: {len(invoices)}")
print(f"Empty text: {(invoices['clean_text'].str.strip() == '').sum()}")
print(f"Avg word count: {invoices['clean_text'].str.split().str.len().mean():.1f}")
print()


# ── 2. Print full text of first 15 invoices ───────────────────────────────
# Read these carefully. Look for patterns in how fields appear.

OUTPUT_DIR = os.path.join(PROJ_ROOT, "data", "exploration")
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "invoice_samples.txt")

with open(out_path, "w") as f:
    for i, (_, row) in enumerate(invoices.head(15).iterrows()):
        f.write(f"\n{'='*70}\n")
        f.write(f"DOCUMENT {i+1} | file: {row.get('file_name', 'unknown')}\n")
        f.write(f"{'='*70}\n")
        f.write(row["clean_text"])
        f.write("\n")

print(f"✓ Saved 15 invoice texts → {out_path}")
print("  → Open this file and read it carefully before writing any regex.\n")


# ── 3. Keyword presence scan ──────────────────────────────────────────────
# How often do useful label keywords appear across all invoices?

keywords = {
    # Invoice number signals
    "invoice no":     r"invoice\s*no",
    "invoice #":      r"invoice\s*#",
    "invoice number": r"invoice\s*number",
    "inv no":         r"inv\.?\s*no",
    "bill no":        r"bill\s*no",
    
    # Date signals
    "invoice date":   r"invoice\s*date",
    "date":           r"\bdate\b",
    "dated":          r"\bdated\b",
    
    # Due date signals
    "due date":       r"due\s*date",
    "payment due":    r"payment\s*due",
    "due":            r"\bdue\b",
    "net 30":         r"net\s*3\d",
    
    # Amount signals
    "total":          r"\btotal\b",
    "amount due":     r"amount\s*due",
    "balance due":    r"balance\s*due",
    "amount":         r"\bamount\b",
    
    # Name signals
    "bill to":        r"bill\s*to",
    "sold to":        r"sold\s*to",
    "to:":            r"\bto\s*:",
    "from:":          r"\bfrom\s*:",
    "attn":           r"\battn\.?\b",
}

print("── Keyword presence across all invoice docs ──────────────────────────")
print(f"{'Keyword':<20} {'Count':>6}  {'%':>6}")
print("-" * 38)
for label, pattern in keywords.items():
    count = invoices["clean_text"].str.contains(pattern, case=False, regex=True).sum()
    pct   = 100 * count / len(invoices)
    print(f"{label:<20} {count:>6}  {pct:>5.1f}%")


# ── 4. Date format scanner ─────────────────────────────────────────────────
# What date formats actually appear in the text?

date_patterns = {
    "MM/DD/YYYY":   r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    "MM/DD/YY":     r"\b\d{1,2}/\d{1,2}/\d{2}\b",
    "DD-Mon-YYYY":  r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
    "Mon DD, YYYY": r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b",
    "YYYY-MM-DD":   r"\b\d{4}-\d{2}-\d{2}\b",
    "DD/MM/YYYY":   r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # overlaps with MM/DD
}

print("\n── Date format patterns found in invoices ────────────────────────────")
for fmt, pat in date_patterns.items():
    matches = invoices["clean_text"].str.extractall(pat)
    print(f"  {fmt:<20}: {len(matches)} occurrences")

# Show actual date examples
all_dates = []
for _, row in invoices.iterrows():
    found = re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b", row["clean_text"])
    all_dates.extend(found[:3])

print(f"\n  Sample date strings found:")
for d in list(set(all_dates))[:20]:
    print(f"    {repr(d)}")


# ── 5. Amount format scanner ───────────────────────────────────────────────

amount_patterns = {
    "$X,XXX.XX":  r"\$[\d,]+\.\d{2}",
    "$X.XX":      r"\$\d+\.\d{2}",
    "bare X.XX":  r"\b\d{1,6}\.\d{2}\b",
    "X,XXX.XX":   r"\b\d{1,3}(?:,\d{3})+\.\d{2}\b",
}

print("\n── Amount format patterns ────────────────────────────────────────────")
for fmt, pat in amount_patterns.items():
    matches = invoices["clean_text"].str.contains(pat, regex=True).sum()
    print(f"  {fmt:<20}: in {matches} documents")

all_amounts = []
for _, row in invoices.iterrows():
    found = re.findall(r"\$[\d,]+\.?\d*|\b\d{1,6}\.\d{2}\b", row["clean_text"])
    all_amounts.extend(found[:3])

print(f"\n  Sample amounts found:")
for a in list(set(all_amounts))[:20]:
    print(f"    {repr(a)}")


# ── 6. OCR noise check ────────────────────────────────────────────────────
# How bad is the noise? Check for common OCR garbling.

noise_indicators = {
    "non-ASCII chars":       invoices["clean_text"].apply(lambda t: bool(re.search(r"[^\x00-\x7F]", t))).sum(),
    "short words ratio>0.5": invoices["clean_text"].apply(
        lambda t: (sum(1 for w in t.split() if len(w) <= 2) / max(len(t.split()), 1)) > 0.5
    ).sum(),
    "avg word length < 3":   invoices["clean_text"].apply(
        lambda t: (sum(len(w) for w in t.split()) / max(len(t.split()), 1)) < 3
    ).sum(),
}

print("\n── OCR noise indicators ──────────────────────────────────────────────")
for k, v in noise_indicators.items():
    print(f"  {k:<30}: {v} docs ({100*v/len(invoices):.1f}%)")

print(f"\n✓ Exploration complete.")
print(f"  Next: open data/exploration/invoice_samples.txt and read 15 full invoice texts.")
print(f"  Then fill in the field pattern table in PHASE_3_FINDINGS.md")