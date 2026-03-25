"""
Phase 3 — Template-Routed Invoice Information Extraction

This builds on the regex baseline by first assigning each document to a
coarse template family, then applying template-specific extraction logic.

Outputs:
  - results/invoice_extractions_template.csv
  - results/invoice_extractions_template.json
  - results/extraction_coverage_template.txt

Usage:
    python scripts/extract_invoices_template.py
"""

import os
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.append(str(BASELINES_DIR))

import re
import json
import csv
from datetime import datetime
import pandas as pd

import extract_invoices as base

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "template"

FIELDS = base.FIELDS
EXTRA_FIELDS = list(base.EXTRA_FIELDS) + ["template_name", "template_score"]


# ---------------------------------------------------------------------------
# Template routing
# ---------------------------------------------------------------------------

def detect_template(text):
    t = text.lower()
    scores = {
        "lab_invoice": 0,
        "law_firm_letterbill": 0,
        "remittance_advice": 0,
        "tabular_invoice": 0,
        "generic_invoice": 0,
    }

    # --- lab / research / Hazleton / Borriston style ---
    for pat in [
        r"\bhazleton\b",
        r"\blaboratories?\b",
        r"\blorillard research center\b",
        r"\battention\b",
        r"\battn\b",
        r"\bleesburg turnpike\b",
    ]:
        if re.search(pat, t):
            scores["lab_invoice"] += 2

    # --- law firm / letter-bill style ---
    for pat in [
        r"\bprofessional corporation\b",
        r"\bwe appreciate the opportunity to serve you\b",
        r"\bthis bill total\b",
        r"\bcurrent professional services\b",
        r"\bdisbursements\b",
        r"\bdue upon receipt\b",
        r"\bre:\b",
    ]:
        if re.search(pat, t):
            scores["law_firm_letterbill"] += 2

    # --- remittance advice style ---
    for pat in [
        r"\bremittance advice\b",
        r"\bclient total\b",
        r"\bremittance total\b",
        r"\breference no\b",
        r"\bvendor code\b",
        r"\binvoice amount\b",
        r"\bamount less\b",
    ]:
        if re.search(pat, t):
            scores["remittance_advice"] += 2

    # --- tabular invoice style ---
    for pat in [
        r"\btotal amount due\b",
        r"\bpayment terms\b",
        r"\bcustomer number\b",
        r"\bunit price\b",
        r"\bquantity\b",
        r"\bdescription\b",
        r"\bsold to\b",
        r"\bshipped to\b",
    ]:
        if re.search(pat, t):
            scores["tabular_invoice"] += 2

    # --- generic invoice backup ---
    for pat in [
        r"\binvoice\b",
        r"\binvoice date\b",
        r"\btotal\b",
        r"\bamount due\b",
        r"\bbill to\b",
        r"\bto:\b",
    ]:
        if re.search(pat, t):
            scores["generic_invoice"] += 1

    best_template = max(scores, key=scores.get)
    best_score = scores[best_template]

    if best_score < 3:
        return "generic_invoice", best_score

    return best_template, best_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_lines(text):
    return [l.strip() for l in str(text).splitlines() if str(l).strip()]


def find_first_matching_line(lines, patterns, max_lines=None):
    subset = lines if max_lines is None else lines[:max_lines]
    for line in subset:
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                return line
    return None


def find_block_after_label(lines, label_patterns, max_lines=5):
    for i, line in enumerate(lines):
        for pat in label_patterns:
            if re.search(pat, line, re.IGNORECASE):
                block = []
                for j in range(i + 1, min(i + 1 + max_lines, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if re.search(
                        r"\b(invoice|date|terms|amount|total|description|qty|quantity|unit price)\b",
                        nxt,
                        re.IGNORECASE,
                    ):
                        break
                    block.append(nxt)
                if block:
                    return block
    return []


def clean_candidate_name(line):
    if not line:
        line = re.sub(r"^(attention|attn)\s*:\s*", "", line, flags=re.IGNORECASE)
        return None
    line = re.split(r"\b(p\.?o\.?\s*box|telephone|telex|fax)\b", line, flags=re.IGNORECASE)[0]
    line = re.sub(r"\s+", " ", line).strip(" ,:-")
    if len(line) < 3:
        return None
    return base._clean_name(line)

def is_bad_recipient_candidate(line):
    if not line:
        return True

    bad_patterns = [
        r"^\s*(remit to|explanation|reference|purchase order|po number|invoice|date|terms|amount|total)\b",
        r"^\s*(ship to|sold to|bill to|to:?)\s*$",
        r"\b(telephone|telex|fax|p\.?o\.?\s*box)\b",
        r"^\d+\s+[A-Za-z]",   # likely street address
        r"\b(street|avenue|road|rd\.?|blvd|boulevard|lane|drive|suite|floor)\b",
    ]

    if len(line.strip()) < 3:
        return True

    if sum(ch.isdigit() for ch in line) > 6:
        return True

    for pat in bad_patterns:
        if re.search(pat, line, re.IGNORECASE):
            return True

    return False


def choose_best_recipient_from_block(block, issuer_name=None):
    best = None
    best_score = -999

    for line in block:
        cand = clean_candidate_name(line)
        if not cand or is_bad_recipient_candidate(cand):
            continue
        if issuer_name and cand == issuer_name:
            continue

        score = 0
        if re.search(r"\b(center|company|inc\.?|corp\.?|ltd\.?|laboratories?|research|associates?|institute)\b", cand, re.IGNORECASE):
            score += 4
        if 2 <= len(cand.split()) <= 6:
            score += 2
        if sum(ch.isdigit() for ch in cand) == 0:
            score += 2
        if re.search(r"\b(lorillard|philip morris|reynolds|tobacco)\b", cand, re.IGNORECASE):
            score += 3
        if re.search(r"\b(street|avenue|road|suite|floor)\b", cand, re.IGNORECASE):
            score -= 4

        if score > best_score:
            best_score = score
            best = cand

    return best

def choose_companyish_line(lines, top_k=10):
    company_pattern = re.compile(
        r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|co\.?|company|companies|"
        r"laboratories?|labs?|associates?|partners?|group|institute|"
        r"industries|international|consulting|services?|research|"
        r"communications?|advertising|corporation|center)\b",
        re.IGNORECASE,
    )

    best = None
    best_score = -999
    for i, line in enumerate(lines[:top_k]):
        if base._is_bad_name_candidate(line):
            continue
        score = 0
        if company_pattern.search(line):
            score += 4
        if i <= 2:
            score += 2
        if 2 <= len(line.split()) <= 9:
            score += 2
        if sum(ch.isdigit() for ch in line) == 0:
            score += 1
        if re.search(r"[A-Za-z]", line):
            score += 1
        if score > best_score:
            best_score = score
            best = line

    return clean_candidate_name(best) if best_score >= 4 else None


def choose_money_from_lines(lines, patterns):
    amount_re = re.compile(r"\$?\s*([\d,]+(?:\.\d{2})?)")
    candidates = []

    for idx, line in enumerate(lines):
        lower = line.lower()
        base_score = 0
        for p, w in patterns:
            if re.search(p, lower):
                base_score += w

        for m in amount_re.finditer(line):
            amt = base._normalise_amount(m.group(1))
            if not amt:
                continue
            score = base_score
            if idx >= int(len(lines) * 0.5):
                score += 1
            candidates.append((score, float(amt), amt, line))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2] if candidates[0][0] >= 2 else None

# ---------------------------------------------------------------------------
# Template-specific extractors
# ---------------------------------------------------------------------------

def extract_lab_invoice(text):
    lines = get_lines(text)
    result = base.extract_all_fields(text)

    # issuer: top block usually strongest
    issuer = choose_companyish_line(lines[:8])
    if issuer:
        result["issuer_name"] = issuer

    # recipient: often after To:, Bill To, or explicit Lorillard line
    block = find_block_after_label(
        lines,
        [r"^to\s*:?", r"\bbill\s*to\b", r"\bsold\s*to\b", r"\battention\b", r"\battn\b"],
        max_lines=5,
    )

    recipient = choose_best_recipient_from_block(
        block,
        issuer_name=result.get("issuer_name")
    )
    if recipient:
        result["recipient_name"] = recipient

    if not result.get("recipient_name"):
        recipient_line = find_first_matching_line(
            lines[:20],
            [r"\blorillard\b", r"\bphilip morris\b", r"\bresearch center\b", r"\bcompany\b"],
        )
        cand = clean_candidate_name(recipient_line)
        if cand and not is_bad_recipient_candidate(cand) and cand != result.get("issuer_name"):
            result["recipient_name"] = cand

    # invoice number: common OCR variants
    if not result.get("invoice_number"):
        for pat in [
            r"(?:invoice|invorce|lnvoice)\s*[#:.-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
            r"\b([A-Z]{1,4}\-?\d{3,10})\b",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m and base._valid_invoice_number(m.group(1).strip()):
                result["invoice_number"] = m.group(1).strip()
                break

    # totals
    total = choose_money_from_lines(
        lines,
        [
            (r"\btotal amount due\b", 6),
            (r"\bamount due\b", 5),
            (r"\btotal\b", 4),
            (r"\bnet amount\b", 4),
        ],
    )
    if total:
        result["total_amount"] = total

    return base.validate_extractions(result)


def extract_law_firm_letterbill(text):
    lines = get_lines(text)
    result = base.extract_all_fields(text)

    # issuer from top letterhead
    issuer = choose_companyish_line(lines[:8])
    if issuer:
        result["issuer_name"] = issuer

    # recipient is usually addressee block between date and body / Re:
    for i, line in enumerate(lines[:25]):
        if re.match(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$", line, re.IGNORECASE):
            block = []
            for j in range(i + 1, min(i + 7, len(lines))):
                nxt = lines[j]
                if re.search(r"^\b(re:|our file no|this bill total|we appreciate)\b", nxt, re.IGNORECASE):
                    break
                if nxt:
                    block.append(nxt)
            for b in block:
                cand = clean_candidate_name(b)
                if cand and cand != result.get("issuer_name"):
                    result["recipient_name"] = cand
                    break
            if result.get("recipient_name"):
                break

    # invoice number / file number
    if not result.get("invoice_number"):
        m = re.search(r"\bour\s*file\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})", text, re.IGNORECASE)
        if m and base._valid_invoice_number(m.group(1)):
            result["invoice_number"] = m.group(1)

    # total from bill total / disbursements
    total = choose_money_from_lines(
        lines,
        [
            (r"\bthis bill total\b", 8),
            (r"\bcurrent disbursements\b", 5),
            (r"\bcurrent professional services\b", 4),
            (r"\btotal\b", 3),
        ],
    )
    if total:
        result["total_amount"] = total

    return base.validate_extractions(result)


def extract_remittance_advice(text):
    lines = get_lines(text)
    result = base.extract_all_fields(text)

    # issuer from very top company line
    issuer = choose_companyish_line(lines[:6])
    if issuer:
        result["issuer_name"] = issuer

    # invoice number often reference/doc number
    if not result.get("invoice_number"):
        for pat in [
            r"\breference\s*no\.?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
            r"\bdocument\s*no\.?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
            r"\binvoice\s*no\.?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m and base._valid_invoice_number(m.group(1)):
                result["invoice_number"] = m.group(1)
                break

    # recipient / client often appears near client line
    for line in lines[:15]:
        if re.search(r"\bclient\b", line, re.IGNORECASE):
            cand = clean_candidate_name(line)
            if cand and cand != result.get("issuer_name"):
                result["recipient_name"] = cand
                break

    # total from remittance total / client total / amount less
    total = choose_money_from_lines(
        lines,
        [
            (r"\bremittance total\b", 8),
            (r"\bclient total\b", 7),
            (r"\binvoice amount\b", 5),
            (r"\bamount less\b", 3),
            (r"\btotal\b", 3),
        ],
    )
    if total:
        result["total_amount"] = total

    return base.validate_extractions(result)


def extract_tabular_invoice(text):
    lines = get_lines(text)
    result = base.extract_all_fields(text)

    issuer = choose_companyish_line(lines[:8])
    if issuer:
        result["issuer_name"] = issuer

    block = find_block_after_label(
        lines,
        [r"\bbill\s*to\b", r"\bsold\s*to\b", r"\bshipped to\b", r"\bship to\b", r"^to\s*:?$"],
        max_lines=5,
    )

    recipient = choose_best_recipient_from_block(
        block,
        issuer_name=result.get("issuer_name")
    )
    if recipient:
        result["recipient_name"] = recipient

    if not result.get("invoice_number"):
        m = re.search(
            r"(?:invoice|invorce|lnvoice)\s*(?:no|number|#)?\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-\/]{2,24})",
            text,
            re.IGNORECASE,
        )
        if m and base._valid_invoice_number(m.group(1)):
            result["invoice_number"] = m.group(1)

    total = choose_money_from_lines(
        lines,
        [
            (r"\btotal amount due\b", 8),
            (r"\bamount due\b", 7),
            (r"\bnet amount\b", 6),
            (r"\bbalance due\b", 6),
            (r"\btotal\b", 4),
        ],
    )
    if total:
        result["total_amount"] = total

    return base.validate_extractions(result)


def extract_generic_invoice(text):
    result = base.extract_all_fields(text)
    return base.validate_extractions(result)


TEMPLATE_EXTRACTORS = {
    "lab_invoice": extract_lab_invoice,
    "law_firm_letterbill": extract_law_firm_letterbill,
    "remittance_advice": extract_remittance_advice,
    "tabular_invoice": extract_tabular_invoice,
    "generic_invoice": extract_generic_invoice,
}


# ---------------------------------------------------------------------------
# Main extraction API
# ---------------------------------------------------------------------------

def extract_all_fields_template(text):
    template_name, template_score = detect_template(text)

    base_result = base.extract_all_fields(text)
    if base_result.get("document_confidence") == "low" and base_result.get("invoice_like_score", 0) <= 0:
        base_result["template_name"] = template_name
        base_result["template_score"] = template_score
        return base_result

    extractor = TEMPLATE_EXTRACTORS.get(template_name, extract_generic_invoice)
    result = extractor(text)
    result["template_name"] = template_name
    result["template_score"] = template_score
    return result


# ---------------------------------------------------------------------------
# Saving / reporting
# ---------------------------------------------------------------------------

def save_csv(results, invoices_df, path):
    rows = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        rows.append({
            "doc_id": row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split": row.get("_split", ""),
            **extraction,
        })

    fieldnames = ["doc_id", "file_name", "_split"] + FIELDS + EXTRA_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(results, invoices_df, path):
    output = []
    for i, extraction in enumerate(results):
        row = invoices_df.iloc[i]
        output.append({
            "doc_id": row.get("doc_id", i),
            "file_name": row.get("file_name", ""),
            "_split": row.get("_split", ""),
            "fields": extraction,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def print_template_breakdown(results):
    counts = {}
    for r in results:
        t = r.get("template_name", "unknown")
        counts[t] = counts.get(t, 0) + 1

    print("\n── Template Breakdown ───────────────────────────────────────────────")
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k:<24} {v:>4}")


def save_coverage_report(coverage, path, results):
    template_counts = {}
    for r in results:
        t = r.get("template_name", "unknown")
        template_counts[t] = template_counts.get(t, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("Phase 3 — Template Extraction Coverage Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  {'Field':<20} {'Found':>6}  {'Total':>6}  {'Coverage':>9}\n")
        f.write("  " + "-" * 50 + "\n")
        for field, stats in coverage.items():
            f.write(
                f"  {field:<20} {stats['found']:>6}  {stats['total']:>6}  {stats['pct']:>7.1f}%\n"
            )

        f.write("\nTemplate breakdown:\n")
        for k, v in sorted(template_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"  - {k}: {v}\n")

        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3 — Template-Routed Invoice Information Extraction")
    print("=" * 60 + "\n")

    invoices = base.load_invoices()

    print("\nExtracting fields from raw text with template routing...")
    results = [extract_all_fields_template(row["raw_text"]) for _, row in invoices.iterrows()]

    coverage = base.compute_coverage(results)
    base.print_coverage_report(coverage)
    print_template_breakdown(results)
    base.print_spot_check(results, invoices, n=5)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "invoice_extractions_template.csv"
    json_path = RESULTS_DIR / "invoice_extractions_template.json"
    cov_path = RESULTS_DIR / "extraction_coverage_template.txt"

    save_csv(results, invoices, csv_path)
    save_json(results, invoices, json_path)
    save_coverage_report(coverage, cov_path, results)

    print(f"\n✓ {csv_path}")
    print(f"✓ {json_path}")
    print(f"✓ {cov_path}")
    print("\nDone. Compare template results against the regex baseline.")


if __name__ == "__main__":
    main()