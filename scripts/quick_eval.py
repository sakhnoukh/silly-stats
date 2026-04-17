"""Quick AFTER-only eval."""
import json, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run import extract_text, clean_text, classify_document, extract_invoice_fields, should_run_invoice_extraction

INVOICES_DIR = Path(__file__).resolve().parent.parent / "tests" / "invoices_real"
GT = json.load(open(INVOICES_DIR.parent / "ground_truth_real6.json", encoding="utf-8"))
FIELDS = ["invoice_number", "invoice_date", "due_date", "issuer_name", "recipient_name", "total_amount"]

def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def strip_accents(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")

def match_str(g, p):
    g, p = _clean(g), _clean(p)
    if not g or not p: return False
    gn, pn = strip_accents(g), strip_accents(p)
    return g in p or p in g or gn in pn or pn in gn

def match_num(g, p):
    g, p = _clean(g).lstrip("0"), _clean(p).lstrip("0")
    return bool(g) and g == p

def match_date(g, p):
    g, p = _clean(g), _clean(p)
    if not g or not p: return False
    if g == p: return True
    gn = re.findall(r"\d+", g); pn = re.findall(r"\d+", p)
    if len(gn) >= 3 and len(pn) >= 3: return sorted(gn[:3]) == sorted(pn[:3])
    return False

def match_amt(g, p):
    def f(s):
        try: return float(re.sub(r"[^\d.]", "", str(s or "")))
        except: return None
    a, b = f(g), f(p)
    if a is None or b is None: return False
    return abs(a - b) <= max(0.01, 0.01 * abs(a))

MF = {
    "invoice_number": match_num, "invoice_date": match_date, "due_date": match_date,
    "issuer_name": match_str, "recipient_name": match_str, "total_amount": match_amt,
}

total_c = total_n = 0
for fp in sorted(INVOICES_DIR.iterdir()):
    if fp.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}:
        continue
    gt = GT.get(fp.name)
    if not gt:
        continue
    raw = extract_text(str(fp))
    if not raw or not raw.strip():
        print(f"  {fp.name:<40} ERROR: no_text")
        continue
    cleaned = clean_text(raw)
    cls = classify_document(raw, cleaned)
    dec = should_run_invoice_extraction(raw, cls)
    if not dec["run_extraction"]:
        n = sum(1 for field in FIELDS if gt.get(field) is not None)
        print(f"  {fp.name:<40} SKIP ({cls['predicted_class']}) 0/{n}")
        total_n += n
        continue
    ext = extract_invoice_fields(raw, file_path=str(fp))
    nc = nt = 0
    wrongs = []
    for field in FIELDS:
        gv = gt.get(field)
        if gv is None:
            continue
        pv = ext.get(field)
        ok = MF[field](gv, pv)
        nt += 1
        nc += int(ok)
        if not ok:
            wrongs.append(f"    {field:<20} gold={gv!r:<30} pred={pv!r}")
    tag = " [OVERRIDE]" if dec["reason"] == "invoice_override" else ""
    print(f"  {fp.name:<40} {nc}/{nt}{tag}")
    for w in wrongs:
        print(w)
    total_c += nc
    total_n += nt

pct = total_c / total_n * 100 if total_n else 0
print(f"\nOVERALL: {total_c}/{total_n} = {pct:.1f}%")
