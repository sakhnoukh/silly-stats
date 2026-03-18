"""
Real Data Integration & Text Extraction Pipeline

Steps:
  1. Cleans up stale pilot data in extracted/, processed/, features/
  2. Copies real receipt images from data/data - receipts/img/ -> data/raw/receipt/
  3. Samples SAMPLES_PER_CLASS images per RVL-CDIP class (email, invoice, form)
     from the existing download manifests
  4. Runs Tesseract OCR on sampled RVL-CDIP images (parallel workers)
  5. Parses SROIE box files for receipts (pre-extracted text, no OCR needed)
  6. Writes data/extracted/dataset_raw.csv for clean_text.py -> build_dataset.py

Usage:
  python3 scripts/extract_real_data.py
"""

import os
import csv
import glob
import shutil
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJ_ROOT, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
EXT_DIR    = os.path.join(DATA_DIR, "extracted")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
FEAT_DIR   = os.path.join(DATA_DIR, "features")

SROIE_IMG_DIR = os.path.join(DATA_DIR, "data - receipts", "img")
SROIE_BOX_DIR = os.path.join(DATA_DIR, "data - receipts", "box")

RVL_CDIP_MANIFEST   = os.path.join(RAW_DIR, "rvl_cdip_manifest.csv")
RVL_FORMS_MANIFEST  = os.path.join(RAW_DIR, "rvl_forms_manifest.csv")

OUTPUT_CSV = os.path.join(EXT_DIR, "dataset_raw.csv")

SAMPLES_PER_CLASS = 200
RANDOM_SEED       = 42
OCR_WORKERS       = 4  # parallel Tesseract processes


# ---------------------------------------------------------------------------
# Step 0 — Clean up stale pilot data
# ---------------------------------------------------------------------------

def cleanup_stale():
    print("[Step 0] Cleaning up stale pilot data …")

    # Wipe extracted subdirs and the CSV
    for item in os.listdir(EXT_DIR):
        p = os.path.join(EXT_DIR, item)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"  removed dir:  extracted/{item}/")
        elif item.endswith(".csv"):
            os.remove(p)
            print(f"  removed file: extracted/{item}")

    # Wipe processed CSVs
    for f in os.listdir(PROC_DIR):
        p = os.path.join(PROC_DIR, f)
        if os.path.isfile(p):
            os.remove(p)
            print(f"  removed file: processed/{f}")

    # Wipe features
    for f in os.listdir(FEAT_DIR):
        p = os.path.join(FEAT_DIR, f)
        if os.path.isfile(p):
            os.remove(p)
            print(f"  removed file: features/{f}")

    # Remove the empty duplicate raw/forms/ dir if present
    forms_dup = os.path.join(RAW_DIR, "forms")
    if os.path.isdir(forms_dup) and not os.listdir(forms_dup):
        os.rmdir(forms_dup)
        print(f"  removed empty dir: raw/forms/")

    # Clear synthetic files from raw/receipt/ (old pilot .txt files)
    rec_raw = os.path.join(RAW_DIR, "receipt")
    os.makedirs(rec_raw, exist_ok=True)
    for f in os.listdir(rec_raw):
        os.remove(os.path.join(rec_raw, f))
        print(f"  removed stale: raw/receipt/{f}")

    print()


# ---------------------------------------------------------------------------
# Step 1 — Copy real receipt images into raw/receipt/
# ---------------------------------------------------------------------------

def copy_receipts():
    print("[Step 1] Copying receipt images -> raw/receipt/ …")
    rec_raw = os.path.join(RAW_DIR, "receipt")
    os.makedirs(rec_raw, exist_ok=True)

    imgs = sorted(os.listdir(SROIE_IMG_DIR))
    copied = 0
    for fname in imgs:
        src = os.path.join(SROIE_IMG_DIR, fname)
        dst = os.path.join(rec_raw, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1

    print(f"  {copied} receipt images in raw/receipt/\n")
    return copied


# ---------------------------------------------------------------------------
# Step 2 — Sample RVL-CDIP images
# ---------------------------------------------------------------------------

def load_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def sample_rvl_class(manifest_path, class_name, n, raw_subdir):
    """
    Load a manifest, filter to class_name, sample n rows,
    return list of (abs_image_path, split, class_name).
    """
    rows = load_manifest(manifest_path)
    class_rows = [r for r in rows if r["class_name"] == class_name]

    random.seed(RANDOM_SEED)
    sampled = random.sample(class_rows, min(n, len(class_rows)))

    result = []
    for r in sampled:
        abs_path = os.path.join(RAW_DIR, r["local_path"])
        result.append((abs_path, r["split"], class_name, r["file_name"]))
    return result


def collect_rvl_samples():
    print(f"[Step 2] Sampling {SAMPLES_PER_CLASS} images per RVL-CDIP class …")
    samples = []

    for class_name, manifest in [
        ("email",   RVL_CDIP_MANIFEST),
        ("invoice", RVL_CDIP_MANIFEST),
        ("form",    RVL_FORMS_MANIFEST),
    ]:
        class_samples = sample_rvl_class(manifest, class_name, SAMPLES_PER_CLASS,
                                         os.path.join(RAW_DIR, class_name))
        print(f"  {class_name}: sampled {len(class_samples)} images")
        samples.extend(class_samples)

    print()
    return samples


# ---------------------------------------------------------------------------
# Step 3 — OCR for RVL-CDIP images
# ---------------------------------------------------------------------------

def ocr_image(args):
    abs_path, split, class_name, file_name = args
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(abs_path)
        text = pytesseract.image_to_string(img).strip()
        return (abs_path, split, class_name, file_name, text, "ocr", True)
    except Exception as e:
        return (abs_path, split, class_name, file_name, "", "ocr_failed", False)


def run_ocr_parallel(samples):
    print(f"[Step 3] Running OCR on {len(samples)} RVL-CDIP images "
          f"({OCR_WORKERS} workers) …")
    results = []
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        futures = {pool.submit(ocr_image, s): s for s in samples}
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                print(f"  OCR: {done}/{len(samples)} done, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    failed = sum(1 for r in results if not r[6])
    print(f"  OCR complete: {len(results)} total, {failed} failed, {elapsed:.0f}s\n")
    return results


# ---------------------------------------------------------------------------
# Step 4 — Parse SROIE box files for receipts
# ---------------------------------------------------------------------------

def parse_box_file(box_path):
    """
    Each line: x1,y1,x2,y2,x3,y3,x4,y4,text
    Concatenate all text tokens into a single string.
    """
    tokens = []
    with open(box_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) == 9:
                tokens.append(parts[8].strip())
    return " ".join(tokens)


def extract_receipts(n):
    print(f"[Step 4] Extracting text from SROIE box files (target: {n} receipts) …")
    box_files = sorted(glob.glob(os.path.join(SROIE_BOX_DIR, "*.csv")))
    rec_raw = os.path.join(RAW_DIR, "receipt")

    random.seed(RANDOM_SEED)
    sampled = random.sample(box_files, min(n, len(box_files)))

    results = []
    for box_path in sampled:
        stem = os.path.splitext(os.path.basename(box_path))[0]
        img_fname = stem + ".jpg"
        text = parse_box_file(box_path)
        results.append((img_fname, "receipt", text))

    ok  = sum(1 for _, _, t in results if len(t) > 20)
    print(f"  {len(results)} receipts extracted, {ok} non-empty\n")
    return results


# ---------------------------------------------------------------------------
# Step 5 — Write extracted/dataset_raw.csv
# ---------------------------------------------------------------------------

def write_csv(rvl_results, receipt_results):
    print("[Step 5] Writing extracted/dataset_raw.csv …")
    os.makedirs(EXT_DIR, exist_ok=True)

    rows = []
    doc_counter = 0

    # ---- RVL-CDIP rows ----
    rvl_source = {
        "email":   "rvl_cdip_email",
        "invoice": "rvl_cdip_invoice",
        "form":    "rvl_cdip_form",
    }
    for abs_path, split, class_name, file_name, text, method, ok in rvl_results:
        doc_counter += 1
        rows.append({
            "doc_id":           f"doc_{doc_counter:04d}",
            "file_name":        file_name,
            "label":            class_name,
            "source_dataset":   rvl_source.get(class_name, "rvl_cdip"),
            "file_type":        ".png",
            "extraction_method": method,
            "raw_text":         text,
        })

        # Save individual .txt mirror
        out_dir = os.path.join(EXT_DIR, class_name)
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    # ---- Receipt rows ----
    for img_fname, label, text in receipt_results:
        doc_counter += 1
        rows.append({
            "doc_id":           f"doc_{doc_counter:04d}",
            "file_name":        img_fname,
            "label":            label,
            "source_dataset":   "sroie_icdar2019",
            "file_type":        ".jpg",
            "extraction_method": "box_file",
            "raw_text":         text,
        })

        out_dir = os.path.join(EXT_DIR, "receipt")
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, os.path.splitext(img_fname)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    fieldnames = ["doc_id", "file_name", "label", "source_dataset",
                  "file_type", "extraction_method", "raw_text"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {len(rows)} rows written -> {OUTPUT_CSV}\n")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Real Data Integration Pipeline")
    print("=" * 60)
    print()

    cleanup_stale()
    copy_receipts()

    rvl_samples   = collect_rvl_samples()
    rvl_results   = run_ocr_parallel(rvl_samples)
    receipt_results = extract_receipts(SAMPLES_PER_CLASS)

    rows = write_csv(rvl_results, receipt_results)

    # --- Summary ---
    from collections import Counter
    label_counts = Counter(r["label"] for r in rows)
    empty_counts = Counter(
        r["label"] for r in rows if len(r["raw_text"].strip()) < 20
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label in sorted(label_counts):
        n     = label_counts[label]
        empty = empty_counts.get(label, 0)
        avg   = sum(len(r["raw_text"]) for r in rows if r["label"] == label) / n
        print(f"  [{label}] {n} docs | avg chars: {avg:.0f} | empty: {empty}")
    print(f"\n  Total: {len(rows)} documents in extracted/dataset_raw.csv")
    print("\nNext: run clean_text.py -> build_dataset.py -> make_features.py")


if __name__ == "__main__":
    main()
