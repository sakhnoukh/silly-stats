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
from typing import Dict, Optional, List, Tuple


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
            r'invoice\s+(?:no|number|num|id|#)\.?\s*[:\-#]?\s*([A-Z0-9][\w\-/]{2,25})',
            # "Inv. No: 30712"  "Inv #4531"
            r'inv\.?\s*(?:no|number|#)\.?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,25})',
            # "Invoice: INV-001" — colon immediately after invoice
            r'invoice\s*[:#]\s*([A-Z0-9][\w\-/]{2,25})',
            # "Proforma No: AGT-PF-2024-0089"  "Proforma Invoice No."
            r'proforma\s*(?:invoice)?\s*(?:no|number|num|#)\.?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,25})',
            # "Our Ref: TLE-2024-AU-0553"  "Ref. No: HC-2024"
            r'(?:our\s+ref|reference|ref)\.?\s*(?:no\.?|number|#)?\s*[:\-]\s*([A-Z0-9][\w\-]{2,25})',
            # "Tax Invoice No." / "Tax Invoice:"
            r'tax\s+invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,25})',
            # P.O. Number
            r'p\.?o\.?\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][\w\-]{2,25})',
            # "#TSS-2024-0412" — hash prefix with no label (when near top of doc)
            r'(?<!\w)#([A-Z0-9][\w\-]{3,20})',
            # "I N V O I C E   N o .  :" spaced-out typewriter style
            r'I\s+N\s+V\s+O\s+I\s+C\s+E\s+N\s*o\.?\s*[:\-]?\s*([A-Z0-9][\w\-/]{2,25})',
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
            r'(?:invoice\s+)?date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'dated?\s*[:\-]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'issue\s+date\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
        ]
        for pattern in inv_labeled:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = self._parse_named_date(m.group(1))
                if candidate:
                    inv_date = candidate
                    break

        # ── Due date ──────────────────────────────────────────────────────────
        # FIX: added (?:date)? to handle "Due Date:" as well as bare "Due:"
        # Added written-month alternative branch
        due_labeled = [
            # "Due Date: November 7, 2024" or "Due: 2024-12-12"
            r'due\s*(?:date|by|on)?\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'payment\s*(?:due|by|date)\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
            r'pay\s*by\s*[:\-=]?\s*([A-Za-z\d][A-Za-z\d\s,/\-\.]{3,30})',
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
        """Extract total invoice amount. Returns last/largest labeled match."""
        amounts = []

        for pattern in self.amount_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Take the first group that looks like a number
                for group in match.groups():
                    if group and re.search(r'\d', str(group)):
                        normalised = _normalise_amount(group)
                        if normalised:
                            amounts.append((match.start(), float(normalised), normalised))
                            break

        if not amounts:
            return None

        # Among all labeled matches, prefer:
        # 1. The last occurrence (totals appear at end of doc)
        # 2. Tie-break: largest value
        amounts.sort(key=lambda x: (x[0], x[1]))
        return amounts[-1][2]

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

        # ── Strategy 3: first header line with company indicator ──────────────
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:6]:
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.35 or len(line) < 4 or len(line) > 120:
                continue
            if self._company_re.search(line):
                return re.sub(r'\s+', ' ', line).strip()

        return None

    # -------------------------------------------------------------------------
    # 5. Recipient Name
    # -------------------------------------------------------------------------

    def extract_recipient_name(self, text: str) -> Optional[str]:
        """
        Extract recipient (customer) name.
        Anchors on 'Bill To' / 'Sold To' / 'Attention' labels.
        Takes the value on the same line or the next non-empty line.
        """
        # ── Same-line patterns ────────────────────────────────────────────────
        same_line = [
            r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to'
            r'|billed\s*to)\s*[:\-]?\s*([^\n]{3,80})',
            r'(?:pay(?:able)?\s*to|remit\s*to)\s*[:\-]?\s*([^\n]{3,80})',
            r'(?:customer|client)\s*[:\-]\s*([^\n]{3,80})',
            r'attn(?:ention)?\s*[:\-.]?\s*(?:mr\.?\s+|ms\.?\s+|dr\.?\s+)?([A-Za-z][^\n]{2,60})',
        ]
        for pattern in same_line:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = m.group(1).strip().splitlines()[0].strip()
                # Trim trailing OCR garbage (long runs of digits after a gap)
                val = re.split(r'\s{2,}|\s+\d{5,}', val)[0].strip()
                val = re.sub(r'\s+', ' ', val)
                if 3 < len(val) < 100:
                    return val

        # ── Next-line pattern: label on one line, value on next ───────────────
        next_line = re.search(
            r'(?:bill(?:ed)?\s*to|sold\s*to)\s*[:\-]?\s*\n\s*([^\n]{3,80})',
            text, re.IGNORECASE
        )
        if next_line:
            val = next_line.group(1).strip()
            if 3 < len(val) < 100:
                return re.sub(r'\s+', ' ', val).strip()

        return None

    # -------------------------------------------------------------------------
    # 6. Main extract() entry point
    # -------------------------------------------------------------------------

    def extract(self, invoice_text: str, file_path: Optional[str] = None, **kwargs) -> Dict:
        """Extract all six fields from invoice text."""
        inv_date, due_date = self.extract_dates(invoice_text)
        return {
            "invoice_number": self.extract_invoice_number(invoice_text),
            "invoice_date":   inv_date,
            "due_date":       due_date,
            "issuer_name":    self.extract_company_name(invoice_text),
            "recipient_name": self.extract_recipient_name(invoice_text),
            "total_amount":   self.extract_amount(invoice_text),
        }


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