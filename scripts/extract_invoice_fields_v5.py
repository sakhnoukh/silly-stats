"""
Phase 3 ÔÇö Invoice Information Extraction

Extracts structured fields from documents classified as invoices:
  - Invoice number
  - Invoice date (YYYY-MM-DD format preferred)
  - Due date (YYYY-MM-DD format preferred)
  - Issuer name (vendor/company)
  - Recipient name (customer/entity)
  - Total amount (numeric, currency)

Uses regex, rule-based extraction, and NLP techniques (no ML models).
Outputs JSON per invoice.

Usage:
  python3 scripts/extract_invoice_fields.py <input_text> [--output output.json]

CHANGELOG (fix/extraction-improvements):
  - Invoice number: label group is now REQUIRED and result must contain a digit.
    Prevents capturing nearby words like "from", "billing", "meridian".
  - Due date: pattern now handles "Due Date:" (with the word "date" between
    "due" and the colon). Multiline matching enabled.
  - Issuer: "Bill To / From" patterns removed from issuer extraction entirely.
    Issuer now anchors to text appearing BEFORE the first "Bill To" block,
    fixing the issuer=recipient confusion on multi-column invoice headers.
"""

import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pytesseract
from PIL import Image

# Windows: set Tesseract path if not already on PATH
import os as _os
if _os.name == "nt":
    _tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _os.path.isfile(_tess):
        pytesseract.pytesseract.tesseract_cmd = _tess

# Lazy-loaded spaCy model (loaded once on first use)
_NLP_MULTI = None
_NLP_EN    = None

_NLP_MULTI = None   # xx_ent_wiki_sm  ÔÇö better for Spanish/French ORG names
_NLP_EN    = None   # en_core_web_trf ÔÇö better for English PERSON names

def _get_nlp_multi():
    global _NLP_MULTI
    if _NLP_MULTI is not None:
        return _NLP_MULTI
    try:
        import spacy
        for m in ["xx_ent_wiki_sm", "en_core_web_trf", "en_core_web_lg"]:
            try:
                _NLP_MULTI = spacy.load(m)
                print(f"[NER-multi] Loaded {m}")
                return _NLP_MULTI
            except OSError:
                continue
    except ImportError:
        pass
    return None

def _get_nlp_en():
    global _NLP_EN
    if _NLP_EN is not None:
        return _NLP_EN
    try:
        import spacy
        for m in ["en_core_web_trf", "en_core_web_lg", "xx_ent_wiki_sm"]:
            try:
                _NLP_EN = spacy.load(m)
                print(f"[NER-en] Loaded {m}")
                return _NLP_EN
            except OSError:
                continue
    except ImportError:
        pass
    return None

# Keep _get_nlp as alias for backward compat
def _get_nlp():
    return _get_nlp_multi()


# ============================================================================
# Field Extraction Patterns & Heuristics
# ============================================================================

def _normalise_amount(amount_str: str) -> Optional[str]:
    """Strip currency symbols, return plain decimal string. Excludes year-like values."""
    if not amount_str:
        return None
    cleaned = re.sub(r'[┬úÔé¼$┬ÑÔé╣\u20ac\s]', '', str(amount_str)).strip()
    if not cleaned:
        return None

    # Detect European format: dot as thousands separator, comma as decimal
    # e.g. "1.871,86" or "871,86"
    eu_match = re.fullmatch(r'(\d{1,3}(?:\.\d{3})*),(\d{1,2})', cleaned)
    if eu_match:
        integer_part = eu_match.group(1).replace('.', '')
        decimal_part = eu_match.group(2)
        cleaned = f"{integer_part}.{decimal_part}"
    else:
        # Standard: strip commas as thousands separators
        cleaned = cleaned.replace(',', '')

    try:
        val = float(cleaned)
        if val <= 0:
            return None
        # Exclude 4-digit years (1900-2099) with no meaningful decimal part
        int_val = int(val)
        if 1900 <= int_val <= 2099 and (val == int_val or abs(val - int_val) < 0.01):
            return None
        return f"{val:.2f}"
    except ValueError:
        return None


class InvoiceExtractor:
    """Regex and rule-based invoice field extractor."""

    def __init__(self):
        # Date patterns (flexible, handles many formats)
        self.date_patterns = [
            r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',
            r'\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        ]

        # Amount patterns ÔÇö ordered most specific ÔåÆ most general
        # Covers: standard labels, non-standard labels, currency after value,
        # spaced-out labels (typewriter), bare totals with no label
        self.amount_patterns = [
            # Most specific multi-word labels
            r'(?:total\s+amount\s+due|amount\s+due|balance\s+due|total\s+due|'
            r'total\s+invoice\s+value|grand\s+total|amount\s+payable|'
            r'please\s+pay|net\s+amount\s+due)\s*[:\-]?\s*'
            r'(?:[┬ú$Ôé¼┬ÑÔé╣\u20ac]|[A-Z]{3})?\s*([\d,\.]+)',
            # "T O T A L :" spaced typewriter style
            r'T\s+O\s+T\s+A\s+L\s*[:\-]?\s*[┬ú$Ôé¼┬ÑÔé╣\u20ac]?\s*([\d,\.]+)',
            # Generic total/amount with currency symbol before value
            r'(?:total|amount|balance)\s*[:]?\s*[┬ú$Ôé¼┬ÑÔé╣\u20ac]\s*([\d,\.]+)',
            # Currency symbol then number (e.g. "Ôé¼ 12,475.10" standalone)
            r'[┬ú$Ôé¼┬ÑÔé╣\u20ac]\s*([\d,]+\.?\d{0,2})',
            # Number then currency code
            r'([\d,]+\.?\d*)\s*(?:usd|eur|gbp|aud|inr|chf)\b',
            # Bare decimal on a "total" line as last resort
            r'(?:total|amount|balance)\s*[:\-]?\s*([\d,]+\.\d{2})',
            # Total label on one line, value on the next (multiline)
            r'(?:total\s+invoice|total|importe\s+total|montant\s+total)\s*\n\s*(?:[┬ú$Ôé¼┬ÑÔé╣\u20ac]\s*)?([\d,\.]+)',
        ]

        # ÔöÇÔöÇ Company indicators used by the issuer heuristic ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self._company_re = re.compile(
            r'\b(inc\.?|corp\.?|ltd\.?|llc\.?|llp\.?|pty\.?|co\.?|company|companies|'
            r'gmbh|ag|s\.l\.?|s\.a\.?|b\.v\.?|n\.v\.?|plc\.?|'
            r'laboratories?|labs?|associates?|partners?|group|institute|'
            r'industries|international|consulting|solutions?|technologies?|'
            r'services?|communications?|newspapers?|advertising|research|'
            r'properties|media|studio|studios?|agency|legal|attorneys?|law|'
            r'engineering|construction|welding|fabrication|supply|trading|'
            r'photography|creative|design|publishing|catering|medical|'
            r'advisory|management|ventures?|holdings?|capital)\b',
            re.IGNORECASE,
        )

        # ÔöÇÔöÇ "Bill To" anchor: marks where recipient block starts ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self._bill_to_re = re.compile(
            r'(?:bill(?:ed)?\s*(?:to|from)|sold\s*to|ship\s*to|billed\s*to'
            r'|invoice\s*to|pay(?:able)?\s*to|remit\s*to)',
            re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # 1. Invoice Number
    # -------------------------------------------------------------------------
    # FIX: label group is now REQUIRED (no longer optional).
    # Captured value must contain at least one digit.
    # Common false-positive words are explicitly rejected.

    def extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number. Label keyword is mandatory; result must have a digit."""

        # Patterns ordered from most to least specific.
        # All require an explicit label before the number.
        labeled_patterns = [
            # "Invoice No: INV-2024-00847"  "Invoice Number: 12345"
            r'invoice\s+(?:no|number|num|id|#)\.?\s*[:\-#]?\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "Inv. No: 30712"  "Inv #4531"
            r'inv\.?\s*(?:no|number|#)\.?\s*[:\-]?\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "Invoice: INV-001" or "Invoice: INV/2023/03/0008"
            r'invoice\s*[:#]\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "Proforma No: AGT-PF-2024-0089"  "Proforma Invoice No."
            r'proforma\s*(?:invoice)?\s*(?:no|number|num|#)\.?\s*[:\-]?\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "Our Ref: TLE-2024-AU-0553"  "Ref. No: HC-2024"
            r'(?:our\s+ref|reference|ref)\.?\s*(?:no\.?|number|#)?\s*[:\-]\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "Tax Invoice No." / "Tax Invoice:"
            r'tax\s+invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9][\w\-/\.]{2,30})',
            # P.O. Number
            r'p\.?o\.?\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][\w\-/\.]{2,30})',
            # "#TSS-2024-0412" or "#BLR_WFLD20151000982590" ÔÇö hash prefix with no label
            r'(?<!\w)#([A-Z0-9][\w\-/]{3,30})',
            # "I N V O I C E   N o .  :" spaced-out typewriter style
            r'I\s+N\s+V\s+O\s+I\s+C\s+E\s+N\s*o\.?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,30})',
            # "Booking ID: IBZY2087" "Confirmation No: 123456" "Folio: 12345"
            r'(?:booking\s*(?:id|no|number|#)|confirmation\s*(?:no|number|#)?|folio)\s*[:\-#]?\s*([A-Z0-9][\w\-/\.]{2,30})',
        ]

        for pattern in labeled_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # Must contain at least one digit
                if not re.search(r'\d', val):
                    continue
                # Must be at least 3 characters
                if len(val) < 3:
                    continue
                # Reject common false positives (words that aren't numbers)
                if re.match(
                    r'^(date|from|to|bill|billing|pay|payment|due|amount|total|'
                    r'subtotal|please|invoice|reference|page|sheet|period|no|number)$',
                    val, re.IGNORECASE
                ):
                    continue
                # Extend with one trailing space-separated numeric token if present
                # (handles "E77148D3 0001" where the number has two parts)
                # Only extend when primary has letters (alphanumeric code) AND
                # the trailing token is NOT a 4-digit year and NOT a date component
                tail_match = re.match(
                    r'\s+(\d{3,6})(?:\s|$)',
                    text[match.end():match.end() + 20],
                )
                if tail_match and re.search(r'[A-Za-z]', val):
                    tail_val = tail_match.group(1)
                    # Skip 4-digit years
                    if not (len(tail_val) == 4 and 1900 <= int(tail_val) <= 2099):
                        val = f"{val} {tail_val}"
                # Strip trailing pure-alpha words (e.g. "039-0002-486391 Leganes" ÔåÆ "039-0002-486391")
                val = re.sub(r'\s+[A-Za-z]+$', '', val).strip()
                if len(val) < 3:
                    continue
                return val.upper()

        return None

    # -------------------------------------------------------------------------
    # 2. Dates (Invoice Date + Due Date)
    # -------------------------------------------------------------------------
    # FIX for due date: pattern now handles "Due Date:" where the word "date"
    # sits between "due" and the colon.  Written-month format added.
    # Multiline flag added so label and value on separate lines still match.

    def extract_dates(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract invoice date and due date.
        Returns (invoice_date, due_date) in YYYY-MM-DD format.
        """
        inv_date = None
        due_date = None

        # ÔöÇÔöÇ Invoice date ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # Try labeled patterns first
        inv_labeled = [
            r'invoice\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'issue\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'billing\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # dated? ÔÇö but NOT when preceded by "due " (which would be "due date")
            r'(?<![Dd][Uu][Ee] )dated?\s*[:\-]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # Bare "Date:" label ÔÇö but only if it's not inside a "Due Date:" context.
            # We match it with a colon required so "Due Date: ..." doesn't match
            # this pattern (because "due date" gets matched by 'dated?' above... no).
            # Simpler fix: require NOT preceded by "due" using a word boundary check.
            r'(?<![Dd][Uu][Ee]\s)(?<![Pp][Aa][Yy]\s)(?:^|(?<=\s)|(?<=\n))date\s*:\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
        ]
        for pattern in inv_labeled:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = self._parse_named_date(m.group(1))
                if candidate:
                    inv_date = candidate
                    break

        # ÔöÇÔöÇ Fallback: standalone written date BEFORE any due-date label ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # Handles law-firm style: date appears in header with no label,
        # e.g. "September 30, 2024" at the top before "Due Date: October 30"
        if not inv_date:
            due_pos_m = re.search(r'due\s*date', text, re.IGNORECASE)
            search_end = due_pos_m.start() if due_pos_m else len(text)
            early_text = text[:search_end]
            month_m = re.search(
                r'\b((?:january|february|march|april|may|june|july|august|'
                r'september|october|november|december)\s+\d{1,2},?\s+\d{4})\b',
                early_text, re.IGNORECASE
            )
            if month_m:
                inv_date = self._parse_named_date(month_m.group(1))

        # ÔöÇÔöÇ Due date ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # FIX: added (?:date)? to handle "Due Date:" as well as bare "Due:"
        # Added written-month alternative branch
        due_labeled = [
            r'due\s*(?:date|by|on)?\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'payment\s*(?:due|by|date)\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'pay(?:ment)?\s*by\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # "Valid Until: December 20, 2024" ÔÇö proforma invoices
            r'valid\s*(?:until|through|to)\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'expir(?:es|y)\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'payment\s+by\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
        ]
        for pattern in due_labeled:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                raw = m.group(1).strip()
                # Skip if this matched the same text as the invoice date
                candidate = self._parse_named_date(raw)
                if candidate and candidate != inv_date:
                    due_date = candidate
                    break

        # ÔöÇÔöÇ Fallback: payment terms string ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # If no explicit due date found, return payment terms as a string
        # For immediate-payment terms, copy invoice_date instead
        if not due_date:
            terms = re.search(
                r'\b(net\s*\d{2,3}(?:\s*days?)?|due\s+on\s+receipt'
                r'|payable\s+immediately|upon\s+receipt'
                r'|cash\s+\d+\s*days?|COD)\b',
                text, re.IGNORECASE
            )
            if terms:
                term_text = terms.group(1).strip().lower()
                if any(kw in term_text for kw in ('receipt', 'immediately')):
                    if inv_date:
                        due_date = inv_date
                    else:
                        due_date = terms.group(1).strip()
                else:
                    due_date = terms.group(1).strip()

        return inv_date, due_date

    def _parse_named_date(self, date_str: str) -> Optional[str]:
        """Parse a date string into YYYY-MM-DD. Returns None if unparseable."""
        if not date_str:
            return None

        date_str = date_str.strip()[:50]

        # Strip stray single characters OCR inserts e.g. "October 23, d 1989"
        date_str = re.sub(r'\s+[a-zA-Z]\s+(?=\d{4})', ' ', date_str)

        # Try numeric formats first
        numeric_match = re.search(
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})'
            r'|(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})',
            date_str
        )
        if numeric_match:
            groups = [g for g in numeric_match.groups() if g is not None]
            result = self._parse_numeric_date(groups)
            if result:
                return result

        # Try written month name
        month_pattern = (
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december|'
            r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
        )
        month_match = re.search(month_pattern, date_str, re.IGNORECASE)
        if month_match:
            month_name = month_match.group(1).lower()
            before = date_str[:month_match.start()]
            after  = date_str[month_match.end():]
            day_match  = re.search(r'(\d{1,2})', before + after)
            year_match = re.search(r'(\d{4})', date_str)
            if day_match and year_match:
                month_map = {
                    'january': 1,  'february': 2,  'march': 3,    'april': 4,
                    'may': 5,      'june': 6,      'july': 7,     'august': 8,
                    'september': 9,'october': 10,  'november': 11,'december': 12,
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                    'oct': 10,'nov': 11,'dec': 12,
                }
                m_num = month_map.get(month_name)
                if m_num:
                    d = int(day_match.group(1))
                    y = int(year_match.group(1))
                    # Sanity check
                    if 1 <= d <= 31 and 1900 <= y <= 2100:
                        return f"{y:04d}-{m_num:02d}-{d:02d}"

        return None

    def _parse_numeric_date(self, parts: List[str]) -> Optional[str]:
        """Parse numeric date components into YYYY-MM-DD."""
        try:
            if len(parts) == 3:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                # Reject impossible values
                if p1 > 31 or p2 > 31:
                    return None
                # YYYY-MM-DD format
                if p1 > 1900:
                    return f"{p1:04d}-{p2:02d}-{p3:02d}"
                # Two-digit year ÔÇö expand
                if p3 < 100:
                    p3 = 2000 + p3 if p3 < 50 else 1900 + p3
                # DD/MM/YYYY vs MM/DD/YYYY ÔÇö prefer DD/MM if day > 12
                if p1 > 12:
                    return f"{p3:04d}-{p2:02d}-{p1:02d}"
                else:
                    return f"{p3:04d}-{p1:02d}-{p2:02d}"
        except Exception:
            pass
        return None

    # -------------------------------------------------------------------------
    # 3. Total Amount
    # -------------------------------------------------------------------------

    def extract_amount(self, text: str) -> Optional[str]:
        """
        Extract total invoice amount.
        Strategy: labeled matches win over bare currency; prefer largest value
        among labeled matches (avoids picking up $0.01 line items).
        """
        labeled_amounts  = []   # from specific total-label patterns
        fallback_amounts = []   # from bare currency symbol patterns

        # Patterns 0-2 and 5 require a total/amount keyword ÔåÆ labeled
        # Patterns 3-4 are bare currency/code ÔåÆ fallback
        # We track this by checking if the pattern text contains a label word
        LABELED_PATTERNS = {0, 1, 2, 5, 6}

        for idx, pattern in enumerate(self.amount_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                for group in match.groups():
                    if group and re.search(r'\d', str(group)):
                        normalised = _normalise_amount(group)
                        if normalised:
                            val = float(normalised)
                            # Minimum plausible invoice total: $0.50
                            if val < 0.50:
                                continue
                            bucket = labeled_amounts if idx in LABELED_PATTERNS else fallback_amounts
                            bucket.append((match.start(), val, normalised))
                            break

        # Prefer labeled matches (require total/amount keyword).
        # Among labeled matches take the LAST occurrence ÔÇö totals appear at
        # the bottom of the document AFTER subtotals, so last = final total.
        # "Largest" was tried but picked subtotals on invoices with discounts.
        if labeled_amounts:
            labeled_amounts.sort(key=lambda x: x[0])
            return labeled_amounts[-1][2]

        # Fallback: last bare currency match
        if fallback_amounts:
            fallback_amounts.sort(key=lambda x: x[0])
            return fallback_amounts[-1][2]

        return None

    # -------------------------------------------------------------------------
    # 4. Issuer Name
    # -------------------------------------------------------------------------
    # FIX: "Bill To/From" patterns completely removed from issuer extraction.
    # New strategy: anchor on "Bill To" and extract company name from ABOVE it.
    # This prevents the header confusion where "Bill To: Apex" was assigned
    # to the issuer field.

    def extract_company_name(self, text: str) -> Optional[str]:
        """
        Extract issuer (vendor) company name.

        Strategy (in order):
        1. Explicit label: "From:", "Vendor:", "Issued by:"
        2. Text appearing BEFORE the first 'Bill To' block ÔÇö company name
           is typically in the header above the billing section.
        3. First substantive header line with company indicators.
        """

        # ÔöÇÔöÇ Strategy 1: explicit label ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        labeled = [
            r'(?:from|vendor|issued\s+by|billed\s+from|seller)\s*[:\-]?\s*([^\n]{4,80})',
        ]
        for pattern in labeled:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                name = m.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if 4 < len(name) < 100:
                    return name

        # ÔöÇÔöÇ Strategy 2: text above the first "Bill To" anchor ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        bill_match = self._bill_to_re.search(text)
        if bill_match:
            above = text[:bill_match.start()]
            lines = [l.strip() for l in above.splitlines() if l.strip()]
            # Scan the last few lines before "Bill To" for a company name
            for line in reversed(lines[-6:]):
                alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
                if alpha_ratio < 0.35 or len(line) < 4 or len(line) > 120:
                    continue
                if self._company_re.search(line):
                    return re.sub(r'\s+', ' ', line).strip()
            # Fallback: first readable line above Bill To
            for line in lines[:4]:
                alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
                if alpha_ratio >= 0.5 and len(line.split()) >= 2:
                    return re.sub(r'\s+', ' ', line).strip()

        # ÔöÇÔöÇ Strategy 3: first header lines ÔÇö company indicator OR person name ÔöÇÔöÇ
        # Collect all plausible candidates, then pick the shortest.
        # Short all-caps / title-case names win over longer taglines.

        # Words that are document-type labels, never issuer names
        _DOC_TYPE_WORDS = {
            'invoice', 'tax invoice', 'proforma invoice', 'proforma',
            'receipt', 'payment receipt', 'sales receipt',
            'quotation', 'quote', 'estimate', 'statement',
            'purchase order', 'delivery note', 'credit note',
            'debit note', 'remittance', 'bill', 'billing statement',
            'retail invoices/bill', 'retail invoices',
        }

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        candidates = []
        for line in lines[:6]:
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.35 or len(line) < 4 or len(line) > 120:
                continue
            # Reject document-type labels
            if line.strip().lower() in _DOC_TYPE_WORDS:
                continue
            words = line.split()
            has_company  = bool(self._company_re.search(line))
            # Person/freelancer: short (Ôëñ3 words), high alpha, all-caps or title-case, no digits
            is_name_line = (1 <= len(words) <= 3 and alpha_ratio >= 0.7
                            and not re.search(r'\d', line)
                            and (line.isupper() or line.istitle()))
            if has_company or is_name_line:
                candidates.append(line)

        if candidates:
            # Shortest wins ÔÇö proper names beat taglines
            best = min(candidates, key=len)
            return re.sub(r'\s+', ' ', best).strip()

        return None

    # -------------------------------------------------------------------------
    # 5. Recipient Name
    # -------------------------------------------------------------------------

    def extract_recipient_name(self, text: str) -> Optional[str]:
        """
        Extract recipient (customer) name.

        More robust than the original version:
        - supports more labels: buyer / recipient / guest / customer name / client name
        - supports same-line and next-line values
        - scans a short recipient block after an anchor
        - allows email recipients when that's what the invoice uses
        """
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        label_re = re.compile(
            r'^(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'pay(?:able)?\s*to|remit\s*to|customer(?:\s*name)?|client(?:\s*name)?|'
            r'buyer(?:\s*name)?|recipient(?:\s*name)?|guest(?:\s*name)?|'
            r'attn(?:ention)?|cliente|comprador|destinatario|acheteur)\b',
            re.IGNORECASE,
        )

        strip_label_re = re.compile(
            r'^(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'pay(?:able)?\s*to|remit\s*to|customer(?:\s*name)?|client(?:\s*name)?|'
            r'buyer(?:\s*name)?|recipient(?:\s*name)?|guest(?:\s*name)?|'
            r'attn(?:ention)?|cliente|comprador|destinatario|acheteur)\s*[:\-]?\s*',
            re.IGNORECASE,
        )

        bad_re = re.compile(
            r'\b(invoice|tax invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'bank|branch|swift|iban|gst|nif|cif|po number|order number|'
            r'email|site|website|www\.|http|tel|phone|mobile|fax|'
            r'method of payment|payment method|'
            r'supplier code|badge|hotel details|check in|check out|room)\b',
            re.IGNORECASE,
        )

        def clean_candidate(val: str) -> Optional[str]:
            if not val:
                return None

            s = strip_label_re.sub("", str(val)).strip()
            s = re.sub(r"\s+", " ", s).strip(" -:|,;")

            if not s:
                return None

            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', s)
            if email_match:
                return email_match.group(0)

            if bad_re.search(s):
                return None

            if re.search(r'\d', s):
                return None

            if len(s) < 2 or len(s) > 80:
                return None

            if re.match(
                r'^(name|customer|client|buyer|recipient|guest|attention|attn|bill to|ship to|sold to|'
                r'bill|sold|code|from|data|details|ship|remit)$',
                s,
                re.IGNORECASE,
            ):
                return None

            return s

        # 1) same-line extraction
        same_line_patterns = [
            r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'pay(?:able)?\s*to|remit\s*to)\s*[:\-]?\s*([^\n]{2,120})',
            r'(?:customer|client|buyer|recipient|guest)(?:\s*name)?\s*[:\-]?\s*([^\n]{2,120})',
            r'attn(?:ention)?\s*[:\-.]?\s*([^\n]{2,120})',
            # Spanish/French labels
            r'(?:cliente|comprador|destinatario|acheteur|facturer\s*[├áa])\s*[:\-]?\s*([^\n]{2,120})',
        ]

        for pattern in same_line_patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = clean_candidate(m.group(1))
                if val:
                    return val

        # 2) anchor block extraction
        for i, line in enumerate(lines):
            if not label_re.search(line):
                continue

            tail = clean_candidate(line)
            if tail and tail.lower() != line.lower():
                return tail

            for nxt in lines[i + 1:i + 4]:
                val = clean_candidate(nxt)
                if val:
                    return val

        # 3) email fallback near recipient-ish labels
        email_block = re.search(
            r'(?:customer|client|buyer|recipient|guest)[^\n]{0,60}\n?([^\n]*@[\w\.-]+\.\w+)',
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if email_block:
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_block.group(1))
            if email_match:
                return email_match.group(0)

        # 4) Spanish/European all-caps personal name fallback
        # e.g. "BEATRIZ MARTIN MARTIN" or "MARTIN MARTIN BEATRIZ"
        for ln in lines[:60]:
            stripped = re.sub(r'\s*\([A-Z ]+\)\s*$', '', ln).strip()
            if not re.fullmatch(r'(?:[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15})(?:\s+[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15}){1,3}', stripped):
                continue
            if bad_re.search(stripped):
                continue
            if re.search(
                r'\b(?:TOTAL|INVOICE|FACTURA|TICKET|RECEIPT|CUSTOMER|DATA|CLIENT|BILL|'
                r'PAYMENT|AMOUNT|DATE|NUMBER|ADDRESS|COUNTRY|CITY|PHONE|EMAIL|TAX|VAT|'
                r'GSTIN|GST|CIN|PAN|IBAN|SWIFT|SRN|HSN|SAC|SUBTOTAL|DISCOUNT|SHIPPING|'
                r'ORIGINAL|COPY|BALANCE|DUE|PAID|REF|QTY|DESCRIPTION)\b',
                stripped, re.I,
            ):
                continue
            return stripped

        return None

    # -------------------------------------------------------------------------
    # OCR helpers for real/translated invoice rescue
    # -------------------------------------------------------------------------

    @staticmethod
    def _post_ocr_cleanup(text: str) -> str:
        """Clean common OCR artefacts from extracted text."""
        # Strip trademark/copyright symbols that break name matching
        text = re.sub(r'[┬«┬®Ôäó]+', '', text)
        # Collapse stray single characters between words (OCR noise)
        text = re.sub(r'(?<=\w) ([A-Za-z]) (?=\w)', r'\1', text)
        # Common OCR confusions in name contexts (pipe ÔåÆ l)
        text = re.sub(r'(?<=[A-Za-z])\|(?=[A-Za-z])', 'l', text)
        return text

    def _file_to_image(self, file_path: str) -> Optional[Image.Image]:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(file_path)
                page = doc[0]
                bm = page.render(scale=3.0)
                return bm.to_pil().convert("RGB")
            except Exception:
                return None

        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            try:
                return Image.open(file_path).convert("RGB")
            except Exception:
                return None

        return None

    # -------------------------------------------------------------------------
    # OCR helpers for real/translated invoice rescue
    # -------------------------------------------------------------------------

    def _ocr_grouped_line_rows(self, file_path: Optional[str]) -> List[Dict[str, object]]:
        if not file_path:
            return []

        img = self._file_to_image(file_path)
        if img is None:
            return []

        try:
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT,
                config='--psm 6',
            )
        except Exception:
            return []

        page_w, page_h = img.size
        split_gap = max(140, int(page_w * 0.08))

        groups = {}
        n = len(data.get("text", []))

        for i in range(n):
            text = str(data["text"][i]).strip()
            if not text:
                continue

            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            right = left + width

            if key not in groups:
                groups[key] = {"items": [], "top": top, "left": left}
            groups[key]["items"].append((left, right, text))
            groups[key]["top"] = min(groups[key]["top"], top)
            groups[key]["left"] = min(groups[key]["left"], left)

        ordered = sorted(groups.values(), key=lambda g: (g["top"], g["left"]))

        rows = []
        row_idx = 0

        for g in ordered:
            items = sorted(g["items"], key=lambda x: x[0])
            if not items:
                continue

            # Split wide multi-column lines into separate segments
            segments = []
            current = [items[0]]

            for item in items[1:]:
                prev_right = current[-1][1]
                gap = item[0] - prev_right
                if gap > split_gap:
                    segments.append(current)
                    current = [item]
                else:
                    current.append(item)

            segments.append(current)

            for seg in segments:
                line = " ".join(t for _, _, t in seg).strip()
                line = re.sub(r"\s+", " ", line).strip()

                if not line:
                    continue

                # Post-OCR cleanup
                line = self._post_ocr_cleanup(line)

                # Drop translation-banner noise
                if re.search(
                    r'(translated from .* to .*|onlinedoc|onlinedoctranslator|translator)',
                    line,
                    re.IGNORECASE,
                ):
                    continue

                rows.append(
                    {
                        "line_idx": row_idx,
                        "top": int(g["top"]),
                        "left": int(seg[0][0]),
                        "text": line,
                    }
                )
                row_idx += 1

        return rows

    def _ocr_grouped_lines(self, file_path: Optional[str]) -> List[str]:
        return [r["text"] for r in self._ocr_grouped_line_rows(file_path)]

    def _looks_bad_recipient(self, val: Optional[str]) -> bool:
        if not val:
            return True

        s = str(val).strip()
        if len(s) < 2:
            return True

        if re.fullmatch(
            r'(bill|bill to|ship to|sold to|customer|customer data|client|buyer|recipient|guest|attn|attention|details|data|code)',
            s,
            re.I,
        ):
            return True

        if re.search(
            r'\b(invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'supplier code|badge|hotel details|check in|check out|room|data|details|'
            r'telephone|phone|tel|customer service)\b',
            s,
            re.I,
        ):
            return True

        if re.fullmatch(r'[A-Z├ü├ë├ì├ô├Ü├£├æ ]+\((?:SPAIN|FRANCE|IRELAND|ECUADOR|GERMANY|NETHERLANDS)\)', s, re.I):
            return True

        return False

    def _looks_bad_issuer(self, val: Optional[str]) -> bool:
        if not val:
            return True
        s = str(val).strip()
        if len(s) < 2:
            return True
        if re.fullmatch(r'(bill|from|vendor|seller|supplier|invoice|neon\s+sign|no\.?|'
                        r'customer|customer\s+data|data|details|code|payment\s+receipt)', s, re.I):
            return True
        # Reject obvious address/facility fragments that aren't companies
        if re.search(
            r'\b(?:business\s+campus|business\s+park|industrial\s+park|shopping\s+centre|'
            r'shopping\s+center|wholesaler|warehouse|logistics\s+centre|logistics\s+center)\b',
            s, re.I,
        ):
            return True
        # Reject 2-3 all-caps words matching the Spanish personal-name shape
        # (these belong to recipient, not issuer)
        if re.fullmatch(
            r'(?:[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15})(?:\s+[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15}){1,2}',
            s,
        ):
            # Only reject if no company indicator present
            if not self._company_re.search(s):
                return True
        if re.search(
            r'(spanish to english|english to spanish|onlinedoc|onlinedoctranslator|'
            r'translation|translated by|^original$|^copy$|www\.|http|@|'
            r'payment\s+receipt)',
            s, re.I,
        ):
            return True
        if re.fullmatch(
            r'(united states|spain|espa├▒a|france|ecuador|ireland|germany|netherlands|'
            r'united kingdom|uk|usa|eu|europe|'
            r'madrid|barcelona|paris|london|dublin|berlin|amsterdam|quito)',
            s, re.I,
        ):
            return True
        # Document title concatenated with client name
        if re.search(r'\b(neon\s+sign|tax\s+invoice|purchase\s+order)\b', s, re.I):
            if len(s.split()) > 4:
                return True
        return False

    def _clean_company_candidate(self, s: str) -> Optional[str]:
        if not s:
            return None

        s = re.sub(
            r'^\s*(?:tax\s+invoice|commercial\s+invoice|invoice|original|copy|bill)\s*[:\-]?\s*',
            '',
            str(s),
            flags=re.IGNORECASE,
        )
        s = re.split(
            r'\b(?:ORIGINAL|Billing information|Invoice recipient|Customer data|Billing address|Mailing address)\b',
            s,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        s = re.sub(r"\s+", " ", s).strip(" -:|,;")

        if not s:
            return None

        if re.search(
            r'(spanish to english|onlinedoc|onlinedoctranslator|translation|translated by|'
            r'www\.|http|@|supplier code|badge|data|details|'
            r'date|due|total|subtotal|amount|balance|vat|tax|phone|tel|fax|bank|swift|iban)',
            s,
            re.IGNORECASE,
        ):
            return None

        if s.lower() in {"bill", "paid", "code", "no"}:
            return None

        if re.fullmatch(r'(united states|spain|espa├▒a|france|ecuador|ireland|germany|netherlands)', s, re.I):
            return None

        digit_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
        if digit_ratio > 0.12:
            return None

        return s


    def _is_strong_company_candidate(self, s: Optional[str]) -> bool:
        if not s:
            return False

        s = str(s).strip()
        if self._looks_bad_issuer(s):
            return False

        words = s.split()
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)

        if self._company_re.search(s):
            return True

        if 1 <= len(words) <= 5 and alpha_ratio >= 0.60 and not re.search(r'\d', s):
            return True

        return False

    def _looks_like_date_string(self, s: str) -> bool:
        if not s:
            return False
        return bool(
            re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', s)
            or re.search(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', s)
            or re.search(
                r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
                r'january|february|march|april|june|july|august|september|october|november|december)\b',
                s,
                re.I,
            )
        )
    
    def _find_date_candidates_in_text(self, s: str) -> List[str]:
        if not s:
            return []

        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        ]

        out = []
        seen = set()

        for pat in patterns:
            for m in re.finditer(pat, s, flags=re.IGNORECASE):
                raw = m.group(0).strip()
                norm = self._parse_named_date(raw)
                if norm and norm not in seen:
                    seen.add(norm)
                    out.append(norm)

        return out

    def _extract_invoice_number_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        anchor_re = re.compile(
            r'(invoice|inv|factura|facture|reference|ref|booking|confirmation|folio)\s*(?:no|number|num|n[o┬║┬░]?|id|#)?',
            re.I,
        )

        bad_re = re.compile(
            r'(total|amount|eur|usd|vat|tax|date|due|balance)',
            re.I,
        )

        def clean_candidate(s: str) -> Optional[str]:
            s = re.sub(r'\s+', ' ', s).strip(' -:|,;')
            if not s:
                return None
            if bad_re.search(s):
                return None
            if self._looks_like_date_string(s):
                return None
            if not re.search(r'\d', s):
                return None
            if len(s) < 3 or len(s) > 40:
                return None
            # Strip trailing pure-alpha words (e.g. "039-0002-486391 Leganes")
            s = re.sub(r'\s+[A-Za-z]+$', '', s).strip()
            if len(s) < 3:
                return None
            return s

        candidates = []

        for i, row in enumerate(rows[:40]):
            text = str(row["text"])

            # same-line anchored extraction
            m = re.search(
                r'(?:invoice|inv|factura|facture|reference|ref|booking|confirmation|folio)\s*(?:no|number|num|n[o┬║┬░]?|id|#)?\s*[:#\-]?\s*([A-Z0-9][A-Z0-9/_\-. ]{2,40})',
                text,
                re.I,
            )
            if m:
                cand = clean_candidate(m.group(1))
                if cand:
                    score = 4
                    if re.search(r'[A-Z]', cand) and re.search(r'\d', cand):
                        score += 1
                    candidates.append((score, int(row["line_idx"]), cand))

            # label line followed by value line
            if anchor_re.search(text):
                for nxt in rows[i + 1:i + 3]:
                    nxt_text = str(nxt["text"])
                    # split into chunks and take the best non-date/non-money token
                    parts = re.split(r'\s{2,}|[|]', nxt_text)
                    if len(parts) == 1:
                        parts = nxt_text.split()

                    joined = clean_candidate(nxt_text)
                    if joined and len(joined.split()) <= 4:
                        candidates.append((3, int(nxt["line_idx"]), joined))

                    for part in parts:
                        cand = clean_candidate(part)
                        if cand:
                            score = 3
                            if re.search(r'[A-Z]', cand) and re.search(r'\d', cand):
                                score += 1
                            candidates.append((score, int(nxt["line_idx"]), cand))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], x[1], -len(x[2])))
        return candidates[0][2]
    
    def _extract_invoice_date_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        anchor_re = re.compile(
            r'(invoice\s*date|issue\s*date|billing\s*date|document\s*issue\s*date|'
            r'date\s*of\s*issue|fecha|fecha\s*de\s*emisi[o├│]n)',
            re.I,
        )

        for i, row in enumerate(rows[:40]):
            text = str(row["text"])
            if not anchor_re.search(text):
                continue

            cands = self._find_date_candidates_in_text(text)
            if cands:
                return cands[0]

            for nxt in rows[i + 1:i + 3]:
                cands = self._find_date_candidates_in_text(str(nxt["text"]))
                if cands:
                    return cands[0]

        return None

    def _extract_due_date_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        anchor_re = re.compile(
            r'(due\s*date|payment\s*due|pay(?:ment)?\s*by|valid\s*until|expiry|'
            r'maturity|vencimiento|fecha\s*l[i├¡]mite|fecha\s*de\s*vencimiento|'
            r'ech[e├®]ance|date\s*d[\' ]ech[e├®]ance)',
            re.I,
        )

        for i, row in enumerate(rows[:40]):
            text = str(row["text"])
            if not anchor_re.search(text):
                continue

            # same line: prefer the last date on a due-date line
            cands = self._find_date_candidates_in_text(text)
            if cands:
                return cands[-1]

            # next lines: also prefer the last date, useful when one row contains both invoice and due dates
            for nxt in rows[i + 1:i + 3]:
                cands = self._find_date_candidates_in_text(str(nxt["text"]))
                if cands:
                    return cands[-1]

        return None

    def _extract_total_amount_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        max_top = max(int(r["top"]) for r in rows)

        strong_total_re = re.compile(
            r'(total\s+amount\s+due|amount\s+due|balance\s+due|grand\s+total|'
            r'total\s+invoice\s+value|net\s+amount\s+due|total\s+to\s+pay|'
            r'total\s+including\s+taxes|total\s+invoice|total\s*\(?eur\)?|'
            r'total\s+fra|importe\s+total|montant\s+total|total\s+ttc|'
            r'total\s+price|total\b)',
            re.I,
        )

        weak_or_bad_re = re.compile(
            r'(unit\s+price|vat\s+excluded|excluding|base|taxes\s+see\s+the\s+rates|'
            r'net\s+amount|subtotal|discount|shipment|price\s+unitary|unitary\s+price|'
            r'vat\s*%|tax\s*%|re\s+quota)',
            re.I,
        )

        def extract_amounts(s: str) -> List[float]:
            vals = re.findall(r'(?:[$Ôé¼┬ú┬Ñ]\s*)?\d[\d.,]{0,20}\d', s, flags=re.I)
            out = []
            for v in vals:
                n = _normalise_amount(v)
                if n:
                    try:
                        out.append(float(n))
                    except Exception:
                        pass
            return out

        candidates = []

        for row in rows:
            text = str(row["text"])
            top = int(row["top"])

            if weak_or_bad_re.search(text):
                continue
            if not strong_total_re.search(text):
                continue

            amounts = extract_amounts(text)
            if not amounts:
                continue

            # Prefer larger amount on strong total lines
            amount = max(amounts)

            score = 0
            if re.search(r'(grand\s+total|amount\s+due|balance\s+due|total\s+to\s+pay|total\s+ttc|importe\s+total)', text, re.I):
                score += 4
            else:
                score += 2

            # Prefer lines in lower half / bottom part of page
            if top > max_top * 0.45:
                score += 2
            if top > max_top * 0.65:
                score += 1

            candidates.append((score, top, amount))

        if not candidates:
            # Last resort: look for decimal monetary amounts in bottom 40% of page.
            # STRICT: must have exactly 2 decimal places and be < 1,000,000
            # to avoid grabbing reference numbers, phone numbers, tax IDs.
            bottom_amounts = []
            for row in rows:
                text = str(row["text"])
                top  = int(row["top"])
                if top < max_top * 0.60:
                    continue
                if weak_or_bad_re.search(text):
                    continue
                # Only match numbers that look like money: digits.2digits
                # \b on left prevents matching tail of longer reference numbers
                for raw in re.findall(r'(?<!\d)(?:[$Ôé¼┬ú┬Ñ]\s*)?\d{1,6}[.,]\d{2}(?!\d)', text):
                    n = _normalise_amount(raw)
                    if n:
                        try:
                            v = float(n)
                            if 1.0 <= v < 1_000_000:
                                bottom_amounts.append(v)
                        except Exception:
                            pass
            if bottom_amounts:
                return f"{max(bottom_amounts):.2f}"
            return None

        candidates.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        return f"{candidates[0][2]:.2f}"

    def _extract_recipient_name_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        label_re = re.compile(
            r'(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer(?:\s*name)?|customer\s*data|client(?:\s*name)?|buyer(?:\s*name)?|'
            r'recipient(?:\s*name)?|guest(?:\s*name)?|invoice\s*recipient|attn(?:ention)?|'
            r'cliente|comprador|destinatario|facturer\s*[├áa]|acheteur|guest\s*name)',
            re.IGNORECASE,
        )

        strip_label_re = re.compile(
            r'^(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer(?:\s*name)?|customer\s*data|client(?:\s*name)?|buyer(?:\s*name)?|'
            r'recipient(?:\s*name)?|guest(?:\s*name)?|invoice\s*recipient|attn(?:ention)?|'
            r'cliente|comprador|destinatario|facturer\s*[├áa]|acheteur|guest\s*name)\s*[:\-]?\s*',
            re.IGNORECASE,
        )

        bad_re = re.compile(
            r'\b(invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'supplier code|badge|hotel details|check in|check out|room|'
            r'bank|swift|iban|gst|nif|cif|data|details|phone|telephone|tel|fax|www\.|http|'
            r'method of payment|payment method|'
            r'customer service)\b',
            re.IGNORECASE,
        )

        def clean(s: str) -> Optional[str]:
            s = strip_label_re.sub("", s).strip()
            s = re.sub(r"\s+", " ", s).strip(" -:|,;")
            if not s:
                return None

            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', s)
            if email_match:
                return email_match.group(0)

            if bad_re.search(s):
                return None
            if re.search(r'\d', s):
                return None
            if re.search(r'\bcustomer\b$', s, re.I):
                return None
            if re.fullmatch(r'(bill|code|sold|from|ship|remit|name)', s, re.I):
                return None
            if len(s) < 2 or len(s) > 80:
                return None
            if re.fullmatch(r'[A-Z├ü├ë├ì├ô├Ü├£├æ ]+\((?:SPAIN|FRANCE|IRELAND|ECUADOR|GERMANY|NETHERLANDS)\)', s, re.I):
                return None

            return s

        # 1) anchored extraction
        for i, row in enumerate(rows):
            line = str(row["text"])
            if not label_re.search(line):
                continue

            same = clean(line)
            if same and same.lower() != line.lower():
                return same

            for nxt in rows[i + 1:i + 4]:
                cand = clean(str(nxt["text"]))
                if cand:
                    return cand

        # 1b) Spanish/European all-caps personal name pattern
        # Matches lines like "BEATRIZ MARTIN MARTIN" or "MARTIN MARTIN BEATRIZ"
        # across top 60% of the page, without needing an anchor label.
        max_top_f = max(int(r["top"]) for r in rows) if rows else 0
        for row in rows[:40]:
            top = int(row["top"])
            if top > max_top_f * 0.60:
                continue
            text = str(row["text"]).strip()
            # Strip trailing parenthetical like "(SPAIN)"
            stripped = re.sub(r'\s*\([A-Z ]+\)\s*$', '', text).strip()
            # Must be 2-4 all-caps words, each 4-15 chars, no digits, ASCII+Spanish
            if not re.fullmatch(r'(?:[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15})(?:\s+[A-Z├ü├ë├ì├ô├Ü├æ├£]{4,15}){1,3}', stripped):
                continue
            if bad_re.search(stripped):
                continue
            if self._is_strong_company_candidate(stripped):
                continue
            if re.search(
                r'\b(?:TOTAL|INVOICE|FACTURA|TICKET|RECEIPT|CUSTOMER|DATA|CLIENT|BILL|'
                r'PAYMENT|AMOUNT|DATE|NUMBER|ADDRESS|COUNTRY|CITY|PHONE|EMAIL|TAX|VAT|'
                r'GSTIN|GST|CIN|PAN|IBAN|SWIFT|SRN|HSN|SAC|SUBTOTAL|DISCOUNT|SHIPPING|'
                r'ORIGINAL|COPY|TOTAL|BALANCE|DUE|PAID|REF|QTY|DESCRIPTION)\b',
                stripped, re.I,
            ):
                continue
            return stripped

        # 2) unlabeled fallback, but only for name-like lines in upper half,
        # and avoid obvious company/header lines
        max_top = max(int(r["top"]) for r in rows) if rows else 0
        candidates = []

        for i, row in enumerate(rows[:25]):
            text = str(row["text"]).strip()
            top = int(row["top"])

            if top > max_top * 0.55:
                continue
            if bad_re.search(text):
                continue
            if self._is_strong_company_candidate(text):
                continue
            if re.search(r'\d', text):
                continue
            if len(text) < 3 or len(text) > 60:
                continue

            words = text.split()
            if not (1 <= len(words) <= 5):
                continue

            next_text = str(rows[i + 1]["text"]) if i + 1 < len(rows) else ""
            has_address_followup = bool(re.search(r'\d', next_text))

            score = 0
            if has_address_followup:
                score += 2
            if text.isupper() or text.istitle():
                score += 1
            if len(words) <= 4:
                score += 1

            if score > 0:
                candidates.append((score, i, text))

        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            return candidates[0][2]

        return None

    def _extract_company_name_ocr(self, file_path: Optional[str]) -> Optional[str]:
        rows = self._ocr_grouped_line_rows(file_path)
        if not rows:
            return None

        recipient_anchor_re = re.compile(
            r'(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer|customer\s*data|client|buyer|recipient|guest|invoice\s*recipient|attn(?:ention)?)',
            re.IGNORECASE,
        )
        seller_anchor_re = re.compile(
            r'(vendor|seller|supplier|from|issued\s+by|billed\s+from|merchant|provider|'
            r'proveedor|emitido\s+por|fournisseur|declared\s+by|sold\s+by|vat\s+has\s+been\s+declared\s+by)',
            re.IGNORECASE,
        )

        # 1) explicit seller-anchor strategy:
        # prefer candidate on the same row to the RIGHT of the anchor, then the next lines
        anchored_candidates = []

        for i, row in enumerate(rows[:35]):
            line = str(row["text"])
            if not seller_anchor_re.search(line):
                continue

            anchor_top = int(row["top"])
            anchor_left = int(row["left"])

            # same-row right-side candidates
            for other in rows:
                if abs(int(other["top"]) - anchor_top) <= 18 and int(other["left"]) > anchor_left + 60:
                    cand = self._clean_company_candidate(str(other["text"]))
                    if cand and self._is_strong_company_candidate(cand):
                        score = 8
                        if self._company_re.search(cand):
                            score += 2
                        anchored_candidates.append((score, cand))

            # same line after stripping label
            stripped = re.sub(
                r'^(?:vendor|seller|supplier|from|issued\s+by|billed\s+from|merchant|provider|'
                r'proveedor|emitido\s+por|fournisseur|declared\s+by|sold\s+by|vat\s+has\s+been\s+declared\s+by)\s*[:\-]?\s*',
                '',
                line,
                flags=re.IGNORECASE,
            ).strip()
            cand = self._clean_company_candidate(stripped)
            if cand and self._is_strong_company_candidate(cand):
                score = 7
                if self._company_re.search(cand):
                    score += 2
                anchored_candidates.append((score, cand))

            # next rows
            for nxt in rows[i + 1:i + 3]:
                cand = self._clean_company_candidate(str(nxt["text"]))
                if cand and self._is_strong_company_candidate(cand):
                    score = 6
                    if self._company_re.search(cand):
                        score += 2
                    anchored_candidates.append((score, cand))

        if anchored_candidates:
            anchored_candidates.sort(key=lambda x: (-x[0], len(x[1])))
            return anchored_candidates[0][1]

        # 2) fallback: scan header before recipient block
        search_rows = rows
        for i, row in enumerate(rows):
            if recipient_anchor_re.search(str(row["text"])):
                search_rows = rows[:max(i, 1)]
                break

        candidates = []
        for idx, row in enumerate(search_rows[:30]):
            cand = self._clean_company_candidate(str(row["text"]))
            if not cand:
                continue
            if not self._is_strong_company_candidate(cand):
                continue

            score = 0
            if self._company_re.search(cand):
                score += 3
            if cand.isupper() or cand.istitle():
                score += 1
            if len(cand.split()) <= 5:
                score += 1
            score += max(0, 3 - idx * 0.2)

            candidates.append((score, cand))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][1]

    # -------------------------------------------------------------------------
    # 6. Main extract() entry point
    # -------------------------------------------------------------------------

    def extract(self, invoice_text: str, file_path: Optional[str] = None, **kwargs) -> Dict:
        """
        Regex-first extractor with OCR-line candidate rescue.
        """
        inv_date, due_date = self.extract_dates(invoice_text)

        result = {
            "invoice_number": self.extract_invoice_number(invoice_text),
            "invoice_date":   inv_date,
            "due_date":       due_date,
            "issuer_name":    self.extract_company_name(invoice_text),
            "recipient_name": self.extract_recipient_name(invoice_text),
            "total_amount":   self.extract_amount(invoice_text),
        }

        if not file_path:
            return result

        # invoice number rescue
                # invoice number rescue
        if (
            not result["invoice_number"]
            or self._looks_like_date_string(str(result["invoice_number"]))
        ):
            cand = self._extract_invoice_number_ocr(file_path)
            if cand:
                result["invoice_number"] = cand

        # invoice date rescue
        if not result["invoice_date"]:
            cand = self._extract_invoice_date_ocr(file_path)
            if cand:
                result["invoice_date"] = cand

        # due date rescue (only if missing)
        if not result["due_date"]:
            cand = self._extract_due_date_ocr(file_path)
            if cand and cand != result["invoice_date"]:
                result["due_date"] = cand

        # total rescue
        ocr_total = self._extract_total_amount_ocr(file_path)
        if ocr_total:
            if not result["total_amount"]:
                result["total_amount"] = ocr_total
            else:
                try:
                    old_val = float(result["total_amount"])
                    new_val = float(ocr_total)
                    replace = False
                    if new_val > old_val * 1.05 and (new_val - old_val) >= 1.0:
                        replace = True
                    elif old_val < 5.00 and new_val > old_val:
                        replace = True
                    if replace:
                        result["total_amount"] = ocr_total
                except Exception:
                    pass

        # recipient rescue
        if self._looks_bad_recipient(result["recipient_name"]):
            cand = self._extract_recipient_name_ocr(file_path)
            if cand and not self._looks_bad_recipient(cand):
                result["recipient_name"] = cand

        # issuer rescue: only replace when regex issuer is bad OR OCR candidate is clearly stronger
        cand = self._extract_company_name_ocr(file_path)
        if cand:
            if self._looks_bad_issuer(result["issuer_name"]):
                result["issuer_name"] = cand
            elif self._is_strong_company_candidate(cand) and not self._is_strong_company_candidate(result["issuer_name"]):
                result["issuer_name"] = cand

        # Clean leading single-char junk from issuer (e.g. "F LEROY MERLIN" ÔåÆ "LEROY MERLIN")
        if result["issuer_name"]:
            result["issuer_name"] = self._strip_issuer_prefix_junk(result["issuer_name"])

        # ÔöÇÔöÇ NER rescue (last resort) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # Use spaCy transformer NER to recover issuer and recipient when
        # OCR+regex still have bad values. Only fires when needed.
        if self._looks_bad_issuer(result["issuer_name"]) or \
           self._looks_bad_recipient(result["recipient_name"]):
            result = self._ner_rescue(invoice_text, result)

        # ÔöÇÔöÇ Email-domain fallback for issuer (last resort) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        # If issuer is still missing/bad and a non-generic email is present,
        # use the second-level domain as the company name (uppercased).
        # Skips generic providers (gmail, hotmail, etc.) and skips emails
        # on lines containing recipient anchors (Bill to:, Ship to:, etc.).
        if self._looks_bad_issuer(result["issuer_name"]):
            recipient_anchor_re = re.compile(
                r'\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|'
                r'customer|client|recipient|attn|delivery\s*address)\b',
                re.IGNORECASE,
            )
            generic = {
                'gmail', 'hotmail', 'yahoo', 'outlook', 'icloud',
                'live', 'aol', 'mail', 'protonmail', 'msn', 'me',
                'gmx', 'inbox', 'fastmail', 'zoho',
            }
            for line in invoice_text.splitlines()[:25]:
                if recipient_anchor_re.search(line):
                    continue
                email_m = re.search(r'[\w\.-]+@([\w-]+)\.[\w\.-]+', line)
                if not email_m:
                    continue
                domain = email_m.group(1).lower()
                if domain in generic or len(domain) < 3:
                    continue
                result["issuer_name"] = domain.upper()
                break

        # Final cleanup: null out values that are still bad after all rescues
        if self._looks_bad_issuer(result["issuer_name"]):
            result["issuer_name"] = None
        if self._looks_bad_recipient(result["recipient_name"]):
            result["recipient_name"] = None

        return result

    def _strip_issuer_prefix_junk(self, s: str) -> str:
        """Remove leading single-char OCR artefacts: 'F LEROY MERLIN' ÔåÆ 'LEROY MERLIN'."""
        if not s:
            return s
        # Leading single letter followed by space + capitalised word
        cleaned = re.sub(r"^[A-Za-z]\s+(?=[A-Z])", "", s).strip()
        # Leading punctuation artefacts like "'Adobe..."
        cleaned = re.sub(r"^['\"`]+", "", cleaned).strip()
        # "Invoice TRANRILOCAR CIA" ÔåÆ "TRANRILOCAR CIA"
        cleaned = re.sub(
            r'^(?:invoice|tax\s+invoice|bill|receipt)\s+',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        return cleaned if len(cleaned) >= 3 else s

    def _ner_rescue(self, text: str, result: Dict) -> Dict:
        """
        Use spaCy NER to recover issuer and recipient from the document text.
        - Multilingual model (xx_ent_wiki_sm) for ORG ÔåÆ issuer (handles Spanish company names)
        - English transformer (en_core_web_trf) for PERSON ÔåÆ recipient (better for English names)
        """
        snippet = text[:2500]

        # Find where recipient block starts
        bill_pos = len(snippet)
        anchor_m = re.search(
            r'bill(?:ed)?\s*to|sold\s*to|customer|client|recipient',
            snippet, re.IGNORECASE
        )
        if anchor_m:
            bill_pos = anchor_m.start()

        def rank_org(pos: int, val: str) -> int:
            """Higher = better issuer candidate."""
            score = 0
            if self._company_re.search(val):
                score += 10
            if 4 <= len(val) <= 60:
                score += 2
            if len(val.split()) <= 5:
                score += 1
            # Prefer earlier position (up to 20 points for position 0)
            score += max(0, 20 - int(pos / 50))
            return score

        # --- Issuer rescue with multilingual model ---
        if self._looks_bad_issuer(result["issuer_name"]):
            nlp_multi = _get_nlp_multi()
            if nlp_multi:
                doc = nlp_multi(snippet)
                orgs = [
                    (e.start_char, e.text.strip())
                    for e in doc.ents
                    if e.label_ == "ORG"
                    and not self._looks_bad_issuer(e.text.strip())
                    and len(e.text.strip()) >= 2
                ]
                pre_bill_orgs = [(p, v) for p, v in orgs if p < bill_pos]
                if pre_bill_orgs:
                    pre_bill_orgs.sort(key=lambda x: -rank_org(x[0], x[1]))
                    result["issuer_name"] = pre_bill_orgs[0][1]
                elif orgs:
                    # Fall back to any ORG ranked by suffix match
                    orgs.sort(key=lambda x: -rank_org(x[0], x[1]))
                    result["issuer_name"] = orgs[0][1]

        # --- Second-pass issuer rescue: scan larger window + OCR text ---
        if self._looks_bad_issuer(result["issuer_name"]):
            nlp_multi = _get_nlp_multi()
            if nlp_multi:
                big_snippet = text[:6000]
                doc = nlp_multi(big_snippet)
                orgs2 = [
                    (e.start_char, e.text.strip())
                    for e in doc.ents
                    if e.label_ == "ORG"
                    and not self._looks_bad_issuer(e.text.strip())
                    and self._company_re.search(e.text.strip())
                ]
                if orgs2:
                    orgs2.sort(key=lambda x: -rank_org(x[0], x[1]))
                    result["issuer_name"] = orgs2[0][1]

        # --- Recipient rescue with English model ---
        if self._looks_bad_recipient(result["recipient_name"]):
            nlp_en = _get_nlp_en()
            if nlp_en:
                doc = nlp_en(snippet)
                persons = [
                    (e.start_char, e.text.strip())
                    for e in doc.ents
                    if e.label_ == "PERSON"
                    and not self._looks_bad_recipient(e.text.strip())
                    and len(e.text.strip()) >= 2
                ]
                orgs_post = [
                    (e.start_char, e.text.strip())
                    for e in doc.ents
                    if e.label_ == "ORG"
                    and e.start_char >= bill_pos
                    and not self._looks_bad_recipient(e.text.strip())
                ]
                if persons:
                    post = [(p, v) for p, v in persons if p >= bill_pos]
                    result["recipient_name"] = (post or persons)[0][1]
                elif orgs_post:
                    result["recipient_name"] = orgs_post[0][1]

        return result


# ============================================================================
# Main Interface
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_invoice_fields.py <input_text_or_file> [--output file.json]")
        sys.exit(1)

    text_input = sys.argv[1]
    output_file = None

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    if os.path.isfile(text_input):
        with open(text_input, "r") as f:
            text = f.read()
    else:
        text = text_input

    extractor = InvoiceExtractor()
    result = extractor.extract(text)

    output = json.dumps(result, indent=2)
    print(output)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\nÔ£ô Saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
