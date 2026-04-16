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

class InvoiceExtractor:
    """Regex and rule-based invoice field extractor."""

    def __init__(self):
        # Date patterns (flexible, handles many formats)
        self.date_patterns = [
            r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b',    # YYYY-MM-DD
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        ]

        # Invoice number patterns
        self.invoice_patterns = [
            r'(?:invoice|inv)\s*(?:no\.?|number|id|#)?\s*[:]?\s*#?([A-Z0-9\-/]+)',
            r'#([A-Z0-9\-/]+)',  # Just #
        ]

        # Amount patterns (currency)
        self.amount_patterns = [
            r'(?:total|amount|balance)\s+(?:amount|due)?\s*[:]?\s*([£$€¥₹])\s*([\d,]+\.?\d*)',
            r'(?:total|amount|balance)\s+(?:amount|due)?\s*[:]?\s*([\d,]+\.?\d*)\s*([£$€¥₹])',
            r'total\s*[:]?\s*([£$€¥₹])\s*([\d,]+(\.\d{2})?)',
            r'(?:total|amount|balance)\s*[:]?\s*([\d,]+\.?\d*)\s*(?:usd|eur|gbp|inr)',
            r'([\d,]+\.\d{2})\s*(?:usd|eur|gbp|inr)',
        ]

        # Company/Name patterns
        self.company_patterns = [
            r'(?:company|vendor|from|issued by)\s*[:]?\s*([^\n]{3,60})',
            r'(?:bill(?:ed)?\s+(?:to|from)|pay(?:able)?\s+(?:to|by))\s*[:]?\s*([^\n]{3,60})',
        ]

        self.recipient_patterns = [
            r'(?:bill(?:ed)?\s+to|customer|invoice to)\s*[:]?\s*([^\n]{3,60})',
            r'(?:pay(?:able)?\s+to)\s*[:]?\s*([^\n]{3,60})',
        ]

    def extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number using regex patterns."""
        text_lower = text.lower()
        
        for pattern in self.invoice_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                inv_no = match.group(1).strip()
                if len(inv_no) > 1:
                    return inv_no
        return None

    def extract_dates(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract invoice date and due date.
        Returns (invoice_date, due_date) in YYYY-MM-DD format.
        """
        # Look for "Date:" "Due:" prefixes first
        inv_date = None
        due_date = None

        # Date: pattern
        date_match = re.search(r'(?:invoice\s+)?date\s*[:=]?\s*([A-Za-z\d\s,/\-\.]+)', text, re.IGNORECASE)
        if date_match:
            inv_date = self._parse_named_date(date_match.group(1))

        # Due date: pattern - more flexible
        due_match = re.search(r'due\s*[:=]?\s*([A-Za-z\d\s,/\-\.]+?)(?=\n|$|Bill|FROM|Acme)', text, re.IGNORECASE)
        if due_match:
            due_str = due_match.group(1).strip()
            due_date = self._parse_named_date(due_str)

        return inv_date, due_date

    def _parse_named_date(self, date_str: str) -> Optional[str]:
        """Parse a date string like 'January 15, 2024' or '2024-01-15'."""
        date_str = date_str.strip()[:50]  # Limit length
        
        # Try numeric formats first
        numeric_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})|(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', date_str)
        if numeric_match:
            groups = [g for g in numeric_match.groups() if g]
            return self._parse_numeric_date(groups)
        
        # Try month name formats
        month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
        month_match = re.search(month_pattern, date_str, re.IGNORECASE)
        if month_match:
            month_name = month_match.group(1).lower()
            # Extract day and year
            day_match = re.search(r'(\d{1,2})', date_str[:month_match.start()] + date_str[month_match.end():])
            year_match = re.search(r'(\d{4})', date_str)
            if day_match and year_match:
                month_map = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12,
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                }
                m = month_map[month_name]
                d = int(day_match.group(1))
                y = int(year_match.group(1))
                return f"{y:04d}-{m:02d}-{d:02d}"
        
        return None

    def _parse_numeric_date(self, parts: List[str]) -> Optional[str]:
        """Parse numeric date components."""
        try:
            if len(parts) == 3:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                # YYYY-MM-DD format
                if p1 > 1900:
                    return f"{p1:04d}-{p2:02d}-{p3:02d}"
                # DD/MM/YYYY or MM/DD/YYYY - prefer DD/MM if day > 12
                if p1 > 12:
                    return f"{p3:04d}-{p2:02d}-{p1:02d}"  # DD/MM format
                else:
                    return f"{p3:04d}-{p1:02d}-{p2:02d}"  # MM/DD format
        except:
            pass
        return None

    def extract_amount(self, text: str) -> Optional[str]:
        """Extract total invoice amount (last occurrence preferred)."""
        text_clean = text
        amounts = []

        for pattern in self.amount_patterns:
            for match in re.finditer(pattern, text_clean, re.IGNORECASE):
                groups = match.groups()
                
                # Pattern 1 & 4: currency then amount (groups: currency, amount, ...)
                # Pattern 2: amount then currency (groups: amount, currency)
                # Pattern 3: just amount (groups: amount)
                
                # Extract the numeric groups (skip non-numeric or currency)
                for i, group in enumerate(groups):
                    if group and any(c.isdigit() for c in group):
                        # Try to extract full number format from this group
                        amount_match = re.search(r'[\d,]+(?:\.\d{1,2})?', group)
                        if amount_match:
                            candidate = amount_match.group(0).replace(',', '')
                            # Prefer multi-digit numbers over decimal-only ("00" vs "500.00")
                            if candidate.lstrip('.').lstrip('0'):  # Has actual digits beyond decimals
                                amounts.append((match.start(), candidate))
                                break  # Take first valid numeric group per match
        
        # Return last (most likely total)
        if amounts:
            amounts.sort(key=lambda x: x[0])
            return amounts[-1][1]
        return None

    def extract_company_name(self, text: str) -> Optional[str]:
        """Extract issuer (vendor) company name."""
        text_lower = text.lower()
        
        for pattern in self.company_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up
                name = re.sub(r'\s+', ' ', name)
                if 3 < len(name) < 100:
                    return name
        return None

    def extract_recipient_name(self, text: str) -> Optional[str]:
        """Extract recipient (customer) name."""
        text_lower = text.lower()
        
        for pattern in self.recipient_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if 3 < len(name) < 100:
                    return name
        return None

    def extract(self, invoice_text: str) -> Dict:
        """Extract all fields from invoice text."""
        return {
            "invoice_number": self.extract_invoice_number(invoice_text),
            "invoice_date": self.extract_dates(invoice_text)[0],
            "due_date": self.extract_dates(invoice_text)[1],
            "issuer_name": self.extract_company_name(invoice_text),
            "recipient_name": self.extract_recipient_name(invoice_text),
            "total_amount": self.extract_amount(invoice_text),
        }


# ============================================================================
# Main Interface
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_invoice_fields.py <input_text_or_file> [--output file.json]")
        print("\nExample:")
        print('  python3 extract_invoice_fields.py "Invoice #123 dated 2024-01-15..."')
        sys.exit(1)

    text_input = sys.argv[1]
    output_file = None

    # Check for output file
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    # Load text (from file or argument)
    if os.path.isfile(text_input):
        with open(text_input, "r") as f:
            text = f.read()
    else:
        text = text_input

    # Extract
    extractor = InvoiceExtractor()
    result = extractor.extract(text)

    # Output
    output = json.dumps(result, indent=2)
    print(output)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\n✓ Saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
