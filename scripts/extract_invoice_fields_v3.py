"""
Phase 3 — Invoice Information Extraction

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


# ============================================================================
# Field Extraction Patterns & Heuristics
# ============================================================================

def _normalise_amount(amount_str: str) -> Optional[str]:
    """Strip currency symbols, return plain decimal string."""
    if not amount_str:
        return None
    cleaned = re.sub(r'[£€$¥₹\u20ac\s]', '', str(amount_str)).replace(',', '').strip()
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if val <= 0:
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

        # Amount patterns — ordered most specific → most general
        # Covers: standard labels, non-standard labels, currency after value,
        # spaced-out labels (typewriter), bare totals with no label
        self.amount_patterns = [
            # Most specific multi-word labels
            r'(?:total\s+amount\s+due|amount\s+due|balance\s+due|total\s+due|'
            r'total\s+invoice\s+value|grand\s+total|amount\s+payable|'
            r'please\s+pay|net\s+amount\s+due)\s*[:\-]?\s*'
            r'(?:[£$€¥₹\u20ac]|[A-Z]{3})?\s*([\d,\.]+)',
            # "T O T A L :" spaced typewriter style
            r'T\s+O\s+T\s+A\s+L\s*[:\-]?\s*[£$€¥₹\u20ac]?\s*([\d,\.]+)',
            # Generic total/amount with currency symbol before value
            r'(?:total|amount|balance)\s*[:]?\s*[£$€¥₹\u20ac]\s*([\d,\.]+)',
            # Currency symbol then number (e.g. "€ 12,475.10" standalone)
            r'[£$€¥₹\u20ac]\s*([\d,]+\.?\d{0,2})',
            # Number then currency code
            r'([\d,]+\.?\d*)\s*(?:usd|eur|gbp|aud|inr|chf)\b',
            # Bare decimal on a "total" line as last resort
            r'(?:total|amount|balance)\s*[:\-]?\s*([\d,]+\.\d{2})',
        ]

        # ── Company indicators used by the issuer heuristic ─────────────────
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

        # ── "Bill To" anchor: marks where recipient block starts ─────────────
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
            # "#TSS-2024-0412" or "#BLR_WFLD20151000982590" — hash prefix with no label
            r'(?<!\w)#([A-Z0-9][\w\-/]{3,30})',
            # "I N V O I C E   N o .  :" spaced-out typewriter style
            r'I\s+N\s+V\s+O\s+I\s+C\s+E\s+N\s*o\.?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,30})',
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

        # ── Invoice date ─────────────────────────────────────────────────────
        # Try labeled patterns first
        inv_labeled = [
            r'invoice\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'issue\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'billing\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # dated? — but NOT when preceded by "due " (which would be "due date")
            r'(?<![Dd][Uu][Ee] )dated?\s*[:\-]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # Bare "Date:" label — but only if it's not inside a "Due Date:" context.
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

        # ── Fallback: standalone written date BEFORE any due-date label ──────
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

        # ── Due date ──────────────────────────────────────────────────────────
        # FIX: added (?:date)? to handle "Due Date:" as well as bare "Due:"
        # Added written-month alternative branch
        due_labeled = [
            r'due\s*(?:date|by|on)?\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'payment\s*(?:due|by|date)\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'pay(?:ment)?\s*by\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            # "Valid Until: December 20, 2024" — proforma invoices
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

        # ── Fallback: payment terms string ──────────────────────────────────
        # If no explicit due date found, return payment terms as a string
        if not due_date:
            terms = re.search(
                r'\b(net\s*\d{2,3}(?:\s*days?)?|due\s+on\s+receipt'
                r'|cash\s+\d+\s*days?|COD)\b',
                text, re.IGNORECASE
            )
            if terms:
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
                # Two-digit year — expand
                if p3 < 100:
                    p3 = 2000 + p3 if p3 < 50 else 1900 + p3
                # DD/MM/YYYY vs MM/DD/YYYY — prefer DD/MM if day > 12
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

        # Patterns 0-2 and 5 require a total/amount keyword → labeled
        # Patterns 3-4 are bare currency/code → fallback
        # We track this by checking if the pattern text contains a label word
        LABELED_PATTERNS = {0, 1, 2, 5}

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
        # Among labeled matches take the LAST occurrence — totals appear at
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
        2. Text appearing BEFORE the first 'Bill To' block — company name
           is typically in the header above the billing section.
        3. First substantive header line with company indicators.
        """

        # ── Strategy 1: explicit label ────────────────────────────────────────
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

        # ── Strategy 2: text above the first "Bill To" anchor ─────────────────
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

        # ── Strategy 3: first header lines — company indicator OR person name ──
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
            # Person/freelancer: short (≤3 words), high alpha, all-caps or title-case, no digits
            is_name_line = (1 <= len(words) <= 3 and alpha_ratio >= 0.7
                            and not re.search(r'\d', line)
                            and (line.isupper() or line.istitle()))
            if has_company or is_name_line:
                candidates.append(line)

        if candidates:
            # Shortest wins — proper names beat taglines
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
            r'attn(?:ention)?)\b',
            re.IGNORECASE,
        )

        strip_label_re = re.compile(
            r'^(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'pay(?:able)?\s*to|remit\s*to|customer(?:\s*name)?|client(?:\s*name)?|'
            r'buyer(?:\s*name)?|recipient(?:\s*name)?|guest(?:\s*name)?|'
            r'attn(?:ention)?)\s*[:\-]?\s*',
            re.IGNORECASE,
        )

        bad_re = re.compile(
            r'\b(invoice|tax invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'bank|branch|swift|iban|gst|nif|cif|po number|order number|'
            r'email|site|website|www\.|http|tel|phone|mobile|fax|'
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
                r'^(name|customer|client|buyer|recipient|guest|attention|attn|bill to|ship to|sold to)$',
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

        return None

    # -------------------------------------------------------------------------
    # OCR helpers for real/translated invoice rescue
    # -------------------------------------------------------------------------

    def _file_to_image(self, file_path: str) -> Optional[Image.Image]:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(file_path)
                page = doc[0]
                bm = page.render(scale=2.0)
                return bm.to_pil().convert("RGB")
            except Exception:
                return None

        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            try:
                return Image.open(file_path).convert("RGB")
            except Exception:
                return None

        return None

    def _ocr_grouped_lines(self, file_path: Optional[str]) -> List[str]:
        if not file_path:
            return []

        img = self._file_to_image(file_path)
        if img is None:
            return []

        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception:
            return []

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

            if key not in groups:
                groups[key] = {"items": [], "top": top, "left": left}
            groups[key]["items"].append((left, text))
            groups[key]["top"] = min(groups[key]["top"], top)
            groups[key]["left"] = min(groups[key]["left"], left)

        ordered = sorted(groups.values(), key=lambda g: (g["top"], g["left"]))

        lines = []
        for g in ordered:
            line = " ".join(t for _, t in sorted(g["items"], key=lambda x: x[0])).strip()
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)

        return lines

    def _looks_bad_recipient(self, val: Optional[str]) -> bool:
        if not val:
            return True

        s = str(val).strip()
        if len(s) < 2:
            return True

        if re.fullmatch(r'(bill to|ship to|sold to|customer|client|buyer|recipient|guest|attn|attention)', s, re.I):
            return True

        if re.search(
            r'\b(invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'supplier code|badge|hotel details|check in|check out|room|data)\b',
            s,
            re.I,
        ):
            return True

        return False

    def _looks_bad_issuer(self, val: Optional[str]) -> bool:
        if not val:
            return True

        s = str(val).strip()
        if len(s) < 2:
            return True

        if re.search(r'(spanish to english|onlinedoc|onlinedoctranslator|^original$|^copy$)', s, re.I):
            return True

        if re.search(r'(www\.|http|@)', s, re.I):
            return True

        if re.fullmatch(r'(united states|spain|españa|france|ecuador|ireland|germany|netherlands)', s, re.I):
            return True

        return False

    def _extract_recipient_name_ocr(self, file_path: Optional[str]) -> Optional[str]:
        lines = self._ocr_grouped_lines(file_path)
        if not lines:
            return None

        label_re = re.compile(
            r'(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer(?:\s*name)?|client(?:\s*name)?|buyer(?:\s*name)?|'
            r'recipient(?:\s*name)?|guest(?:\s*name)?|attn(?:ention)?)',
            re.IGNORECASE,
        )

        strip_label_re = re.compile(
            r'^(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer(?:\s*name)?|client(?:\s*name)?|buyer(?:\s*name)?|'
            r'recipient(?:\s*name)?|guest(?:\s*name)?|attn(?:ention)?)\s*[:\-]?\s*',
            re.IGNORECASE,
        )

        bad_re = re.compile(
            r'\b(invoice|date|due|subtotal|total|amount|balance|vat|tax|'
            r'supplier code|badge|hotel details|check in|check out|room|'
            r'bank|swift|iban|gst|nif|cif|data|phone|tel|fax|www\.|http)\b',
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

            if len(s) < 2 or len(s) > 80:
                return None

            return s

        for i, line in enumerate(lines):
            if not label_re.search(line):
                continue

            same = clean(line)
            if same and same.lower() != line.lower():
                return same

            for nxt in lines[i + 1:i + 4]:
                cand = clean(nxt)
                if cand:
                    return cand

        return None

    def _extract_company_name_ocr(self, file_path: Optional[str]) -> Optional[str]:
        lines = self._ocr_grouped_lines(file_path)
        if not lines:
            return None

        recipient_anchor_re = re.compile(
            r'(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|billed\s*to|'
            r'customer|client|buyer|recipient|guest|attn(?:ention)?)',
            re.IGNORECASE,
        )

        bad_re = re.compile(
            r'(spanish to english|onlinedoc|onlinedoctranslator|^original$|^copy$|'
            r'www\.|http|@|supplier code|badge|data|date|due|total|subtotal|'
            r'amount|balance|vat|tax|phone|tel|fax|bank|swift|iban)',
            re.IGNORECASE,
        )

        search_lines = lines
        for i, line in enumerate(lines):
            if recipient_anchor_re.search(line):
                search_lines = lines[:max(i, 1)]
                break

        candidates = []
        for raw in search_lines[:12]:
            s = re.sub(
                r'^\s*(?:tax\s+invoice|commercial\s+invoice|invoice|original)\s*[:\-]?\s*',
                '',
                raw,
                flags=re.IGNORECASE,
            ).strip()
            s = re.sub(r"\s+", " ", s).strip(" -:|,;")

            if not s:
                continue
            if bad_re.search(s):
                continue
            if re.fullmatch(r'(united states|spain|españa|france|ecuador|ireland|germany|netherlands)', s, re.I):
                continue
            if re.search(r'\d', s):
                continue

            words = s.split()
            alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
            has_company = bool(self._company_re.search(s))
            is_name_like = (
                1 <= len(words) <= 5
                and alpha_ratio >= 0.60
                and (s.isupper() or s.istitle())
            )

            if has_company or is_name_like:
                candidates.append(s)

        if not candidates:
            return None

        candidates.sort(key=lambda s: (0 if self._company_re.search(s) else 1, len(s)))
        return candidates[0]

    # -------------------------------------------------------------------------
    # 6. Main extract() entry point
    # -------------------------------------------------------------------------

    def extract(self, invoice_text: str, file_path: Optional[str] = None, **kwargs) -> Dict:
        """
        Regex-first extractor with OCR-line rescue for issuer/recipient.
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

        # OCR rescue only when a real file is available and the regex result
        # is missing or obviously polluted.
        if file_path:
            if self._looks_bad_recipient(result["recipient_name"]):
                cand = self._extract_recipient_name_ocr(file_path)
                if cand:
                    result["recipient_name"] = cand

            if self._looks_bad_issuer(result["issuer_name"]):
                cand = self._extract_company_name_ocr(file_path)
                if cand:
                    result["issuer_name"] = cand

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
        print(f"\n✓ Saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()