"""
Phase 3 — Build Candidate Tables for ML Extraction

Creates candidate-level datasets for classical ML ranking.
Each candidate row corresponds to one possible value for one field.

Current target fields:
  - recipient_name
  - invoice_number
  - total_amount

Outputs:
  - results/candidates_recipient_name.csv
  - results/candidates_invoice_number.csv
  - results/candidates_total_amount.csv

Usage:
    python scripts/build_extraction_candidates.py
"""

import os
import sys
import re
import csv
from pathlib import Path
from collections import Counter

import pandas as pd

from pathlib import Path
from scripts.phase_3_extraction.baselines import extract_invoices as base

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "candidates"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_text(x):
    return "" if pd.isna(x) else str(x)


def get_lines(text):
    return [line.strip() for line in safe_text(text).splitlines() if line.strip()]


def rel_pos(idx, n):
    if n <= 1:
        return 0.0
    return round(idx / (n - 1), 4)


def count_digits(s):
    return sum(ch.isdigit() for ch in s)


def count_alpha(s):
    return sum(ch.isalpha() for ch in s)


def uppercase_ratio(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    return round(sum(ch.isupper() for ch in letters) / len(letters), 4)


def digit_ratio(s):
    if not s:
        return 0.0
    return round(count_digits(s) / max(len(s), 1), 4)


def token_count(s):
    return len(str(s).split())


def has_org_suffix(s):
    return int(bool(re.search(
        r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|company|companies|corporation|group|institute|"
        r"research|center|centre|services?|laboratories?|labs?|associates?)\b",
        s,
        re.IGNORECASE,
    )))


def has_address_word(s):
    return int(bool(re.search(
        r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd|lane|drive|dr\.?|suite|floor)\b",
        s,
        re.IGNORECASE,
    )))


def has_label_word(s):
    return int(bool(re.search(
        r"\b(invoice|date|terms|amount|total|reference|explanation|remit|payment|purchase order)\b",
        s,
        re.IGNORECASE,
    )))

def is_bad_recipient_candidate_text(s):
    s = safe_text(s).strip()
    if not s:
        return True

    bad_patterns = [
        r"^\s*(remit to|client copy|explanation|reference|purchase order|po number|invoice|date|terms|amount|total)\b",
        r"^\s*(attention|attn)\s*:?\s*$",
        r"^\s*(accounting|issued|number|final)\s*$",
        r"\b(net\s+\d+|cash\s+\d+\s+days|due upon receipt)\b",
        r"^\s*\W*\s*$",
    ]

    if len(s) < 4:
        return True
    if token_count(s) > 10:
        return True
    if count_digits(s) > 6:
        return True

    for pat in bad_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return True

    return False

def is_bad_invoice_number_candidate_text(s):
    s = safe_text(s).strip()
    if not s:
        return True

    bad_exact = {
        "date", "amount", "number", "purchase", "final",
        "issued", "invoice", "reference", "total"
    }

    if s.lower() in bad_exact:
        return True

    if len(s) < 4 or len(s) > 24:
        return True

    if not re.search(r"\d", s):
        return True

    if re.fullmatch(r"\d{4}", s):
        return True  # likely year, not invoice number

    if re.fullmatch(r"[A-Z]?\d{3,4}", s):
        return True  # too weak / ambiguous

    if re.search(r"[^\w\-\/]", s):
        return True

    return False

def word_shape(s):
    """
    Simplified word-shape feature for generalization.
    Example:
      'INV-1234' -> 'AAA-0000'
      'Lorillard' -> 'Aaaaaaaaa'
    """
    out = []
    for ch in str(s):
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    return "".join(out)[:40]


def context_flags(line, prev_line="", next_line=""):
    joined = " | ".join([safe_text(prev_line), safe_text(line), safe_text(next_line)])
    joined_l = joined.lower()

    return {
        "ctx_has_invoice": int(bool(re.search(r"\binvoice\b", joined_l))),
        "ctx_has_invoice_no": int(bool(re.search(r"\binvoice\s*(?:no|number|#)\b", joined_l))),
        "ctx_has_bill_to": int(bool(re.search(r"\bbill\s*to\b", joined_l))),
        "ctx_has_sold_to": int(bool(re.search(r"\bsold\s*to\b", joined_l))),
        "ctx_has_ship_to": int(bool(re.search(r"\bship(?:ped)?\s*to\b", joined_l))),
        "ctx_has_attention": int(bool(re.search(r"\b(attn|attention)\b", joined_l))),
        "ctx_has_total": int(bool(re.search(r"\btotal\b", joined_l))),
        "ctx_has_amount_due": int(bool(re.search(r"\b(amount due|balance due|net amount|total amount due)\b", joined_l))),
        "ctx_has_reference": int(bool(re.search(r"\breference\b", joined_l))),
        "ctx_has_due": int(bool(re.search(r"\bdue\b", joined_l))),
    }


def document_features(text):
    return {
        "invoice_like_score": base.compute_invoice_like_score(text),
        "ocr_quality_score": base.compute_ocr_quality_score(text),
        "doc_num_lines": len(get_lines(text)),
        "doc_num_chars": len(safe_text(text)),
    }


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def generate_recipient_candidates(text):
    """
    Generalizable recipient candidates:
    - lines after bill-to / sold-to / ship-to / to / attention labels
    - organization-like lines in first 25 lines
    - avoid overfitting to dataset-specific names
    """
    lines = get_lines(text)
    n = len(lines)
    candidates = []

    label_patterns = [
        r"\bbill\s*to\b",
        r"\bsold\s*to\b",
        r"\bship(?:ped)?\s*to\b",
        r"^to\s*:?\s*$",
        r"\battn\b",
        r"\battention\b",
    ]

    # 1) block-after-label candidates
    for i, line in enumerate(lines):
        if any(re.search(p, line, re.IGNORECASE) for p in label_patterns):
            for j in range(i + 1, min(i + 6, n)):
                cand = lines[j].strip()
                if not cand:
                    break
                if re.search(r"\b(invoice|date|terms|amount|total|qty|quantity|description)\b", cand, re.IGNORECASE):
                    break
                candidates.append(("after_label", j, cand))

    # 2) early organization-like lines
    for i, line in enumerate(lines[:25]):
        if 2 <= token_count(line) <= 10 and count_digits(line) <= 6:
            if has_org_suffix(line) or re.search(r"\b(to|client|customer)\b", line, re.IGNORECASE):
                candidates.append(("early_org_line", i, line))

    # 3) attention lines themselves
    for i, line in enumerate(lines[:30]):
        if re.search(r"\b(attn|attention)\b", line, re.IGNORECASE):
            candidates.append(("attention_line", i, line))

    # deduplicate + filter while preserving first occurrence
    seen = set()
    deduped = []
    for source, idx, cand in candidates:
        cand = cand.strip()
        key = cand.lower()
        if not key:
            continue
        if is_bad_recipient_candidate_text(cand):
            continue
        if key not in seen:
            seen.add(key)
            deduped.append((source, idx, cand))

    return deduped


def generate_invoice_number_candidates(text):
    """
    Generalizable invoice number candidates:
    - labeled patterns
    - alphanumeric tokens near invoice/reference/file/doc labels
    """
    lines = get_lines(text)
    candidates = []

    strong_patterns = [
        r"(?:invoice|invorce|lnvoice|invoace)\s*(?:no|number|num|#)?\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\breference\s*(?:no|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bour\s*file\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bdoc(?:ument)?\s*(?:no|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        r"\bestimate\s*#\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
    ]

    tokenish_pattern = re.compile(r"\b[A-Z0-9][A-Z0-9\-\/]{3,24}\b", re.IGNORECASE)

    for i, line in enumerate(lines[:40]):
        # labeled candidates
        for pat in strong_patterns:
            for m in re.finditer(pat, line, re.IGNORECASE):
                candidates.append(("labeled", i, m.group(1).strip()))

        # nearby token candidates on invoice-ish lines
        if re.search(r"\b(invoice|reference|file|document|doc)\b", line, re.IGNORECASE):
            for m in tokenish_pattern.finditer(line):
                tok = m.group(0).strip()
                if base._valid_invoice_number(tok):
                    candidates.append(("invoiceish_line_token", i, tok))

    seen = set()
    deduped = []
    for source, idx, cand in candidates:
        cand = cand.strip()
        key = cand.lower()
        if not key:
            continue
        if is_bad_invoice_number_candidate_text(cand):
            continue
        if key not in seen:
            seen.add(key)
            deduped.append((source, idx, cand))

    return deduped

def generate_total_amount_candidates(text):
    """
    Generalizable total candidates:
    - all amount-like strings
    - especially from total-like lines
    """
    lines = get_lines(text)
    candidates = []
    amount_re = re.compile(r"\$?\s*([\d,]+(?:\.\d{2})?)")

    for i, line in enumerate(lines):
        lower = line.lower()
        for m in amount_re.finditer(line):
            raw_amt = m.group(1).strip()
            norm_amt = base._normalise_amount(raw_amt)
            if not norm_amt:
                continue

                        # reject tiny amounts unless on a strong total-like line
            try:
                amt_value = float(norm_amt)
            except Exception:
                amt_value = 0.0

            if amt_value < 10 and not re.search(
                r"\b(total|amount due|balance due|net amount|this bill total)\b", lower
            ):
                continue

            # keep all decimals; keep integer-like only on total-ish lines
            if "." not in raw_amt and not re.search(r"\b(total|amount due|balance due|net amount|this bill total)\b", lower):
                continue

            source = "money_anywhere"
            if re.search(r"\b(total amount due|amount due|balance due|net amount|this bill total)\b", lower):
                source = "money_total_line"
            elif re.search(r"\btotal\b", lower):
                source = "money_totalish_line"

            candidates.append((source, i, norm_amt))

    seen = set()
    deduped = []
    for source, idx, cand in candidates:
        key = cand
        if key not in seen:
            seen.add(key)
            deduped.append((source, idx, cand))

    return deduped


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_common_candidate_features(field, candidate, source, line_idx, lines, text):
    n = len(lines)
    prev_line = lines[line_idx - 1] if line_idx > 0 else ""
    line = lines[line_idx] if 0 <= line_idx < n else ""
    next_line = lines[line_idx + 1] if line_idx + 1 < n else ""

    doc_feats = document_features(text)
    ctx = context_flags(line, prev_line, next_line)

    feats = {
        "field": field,
        "candidate_text": candidate,
        "candidate_source": source,
        "line_idx": line_idx,
        "line_rel_pos": rel_pos(line_idx, n),
        "line_token_count": token_count(line),
        "candidate_len": len(candidate),
        "candidate_token_count": token_count(candidate),
        "candidate_digit_count": count_digits(candidate),
        "candidate_alpha_count": count_alpha(candidate),
        "candidate_digit_ratio": digit_ratio(candidate),
        "candidate_uppercase_ratio": uppercase_ratio(candidate),
        "candidate_has_org_suffix": has_org_suffix(candidate),
        "candidate_has_address_word": has_address_word(candidate),
        "candidate_has_label_word": has_label_word(candidate),
        "candidate_shape": word_shape(candidate),
        "line_has_colon": int(":" in line),
        "line_is_upper": int(line.isupper()),
        "line_digit_ratio": digit_ratio(line),
        "line_has_dollar": int("$" in line),
        "line_has_org_suffix": has_org_suffix(line),
        "line_has_address_word": has_address_word(line),
        "line_has_label_word": has_label_word(line),
        "is_top_10_lines": int(line_idx < 10),
        "is_top_20_lines": int(line_idx < 20),
        "is_bottom_half": int(line_idx >= max(1, n // 2)),
        "is_bottom_quarter": int(line_idx >= max(1, int(0.75 * n))),
        **ctx,
        **doc_feats,
    }
    return feats


def build_recipient_features(candidate, source, line_idx, lines, text):
    feats = build_common_candidate_features(
        "recipient_name", candidate, source, line_idx, lines, text
    )
    feats.update({
        "recipient_like_person_prefix": int(bool(re.search(r"^(mr|mrs|ms|dr)\.?\b", candidate, re.IGNORECASE))),
        "recipient_like_attention": int(bool(re.search(r"\b(attn|attention)\b", candidate, re.IGNORECASE))),
        "recipient_looks_address_only": int(
            has_address_word(candidate) and not has_org_suffix(candidate)
        ),
        "recipient_digit_heavy": int(count_digits(candidate) >= 5),
    })
    return feats


def build_invoice_number_features(candidate, source, line_idx, lines, text):
    feats = build_common_candidate_features(
        "invoice_number", candidate, source, line_idx, lines, text
    )
    feats.update({
        "inv_has_digit": int(bool(re.search(r"\d", candidate))),
        "inv_has_dash": int("-" in candidate),
        "inv_has_slash": int("/" in candidate),
        "inv_all_capsish": int(candidate.upper() == candidate),
        "inv_valid_by_rules": int(base._valid_invoice_number(candidate)),
        "inv_short_code": int(bool(re.match(r"^[A-Z]?\d{3,4}$", candidate))),
    })
    return feats


def build_total_amount_features(candidate, source, line_idx, lines, text):
    feats = build_common_candidate_features(
        "total_amount", candidate, source, line_idx, lines, text
    )
    try:
        amt = float(candidate)
    except Exception:
        amt = -1.0

    line = lines[line_idx] if 0 <= line_idx < len(lines) else ""
    feats.update({
        "amt_value": amt,
        "amt_ge_100": int(amt >= 100),
        "amt_ge_1000": int(amt >= 1000),
        "amt_line_has_total": int(bool(re.search(r"\btotal\b", line, re.IGNORECASE))),
        "amt_line_has_amount_due": int(bool(re.search(r"\b(amount due|balance due|net amount|this bill total)\b", line, re.IGNORECASE))),
        "amt_line_dot_leader_heavy": int(line.count(".") > 8),
    })
    return feats


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_candidate_rows_for_doc(row):
    text = safe_text(row.get("raw_text", ""))
    lines = get_lines(text)

    common_meta = {
        "doc_id": row.get("doc_id", ""),
        "file_name": row.get("file_name", ""),
        "_split": row.get("_split", ""),
    }

    recipient_rows = []
    for source, idx, cand in generate_recipient_candidates(text):
        feats = build_recipient_features(cand, source, idx, lines, text)
        recipient_rows.append({**common_meta, **feats})

    invoice_number_rows = []
    for source, idx, cand in generate_invoice_number_candidates(text):
        feats = build_invoice_number_features(cand, source, idx, lines, text)
        invoice_number_rows.append({**common_meta, **feats})

    total_amount_rows = []
    for source, idx, cand in generate_total_amount_candidates(text):
        feats = build_total_amount_features(cand, source, idx, lines, text)
        total_amount_rows.append({**common_meta, **feats})

    return recipient_rows, invoice_number_rows, total_amount_rows


def save_csv(rows, path):
    if not rows:
        print(f"  ⚠️  No rows to save for {path}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")


def print_summary(name, rows):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  candidate rows: {len(rows)}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    docs = df["doc_id"].nunique()
    avg_per_doc = round(len(df) / max(docs, 1), 2)
    print(f"  docs covered: {docs}")
    print(f"  avg candidates/doc: {avg_per_doc}")

    if "candidate_source" in df.columns:
        print("  source breakdown:")
        counts = Counter(df["candidate_source"])
        for k, v in counts.most_common():
            print(f"    - {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3 — Build Extraction Candidate Tables")
    print("=" * 60 + "\n")

    invoices = base.load_invoices()

    recipient_rows = []
    invoice_number_rows = []
    total_amount_rows = []

    print("Building candidates from raw_text...")
    for _, row in invoices.iterrows():
        rec_rows, inv_rows, amt_rows = build_candidate_rows_for_doc(row)
        recipient_rows.extend(rec_rows)
        invoice_number_rows.extend(inv_rows)
        total_amount_rows.extend(amt_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rec_path = RESULTS_DIR / "candidates_recipient_name.csv"
    inv_path = RESULTS_DIR / "candidates_invoice_number.csv"
    amt_path = RESULTS_DIR / "candidates_total_amount.csv"

    save_csv(recipient_rows, rec_path)
    save_csv(invoice_number_rows, inv_path)
    save_csv(total_amount_rows, amt_path)

    print_summary("Recipient candidates", recipient_rows)
    print_summary("Invoice-number candidates", invoice_number_rows)
    print_summary("Total-amount candidates", total_amount_rows)

    print(f"\n✓ {rec_path}")
    print(f"✓ {inv_path}")
    print(f"✓ {amt_path}")
    print("\nDone. Next step: create a labeled subset and train rankers.")


if __name__ == "__main__":
    main()