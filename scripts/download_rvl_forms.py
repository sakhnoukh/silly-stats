"""
Download forms (label=1) from RVL-CDIP.

Strategy (most storage-efficient):
  1. Download the 3 tiny label-metadata files (~17 MB total)
  2. Parse them to build a set of target image paths (forms only)
  3. Stream the 38 GB tar.gz archive over HTTP, extracting ONLY matching
     images — the full archive is never saved to disk

Data URLs (hosted on HuggingFace under the canonical rvl_cdip repo):
  - train.txt / test.txt / val.txt  →  image_path  label_id
  - rvl-cdip.tar.gz                 →  TIFF images under images/

Saves images to:
  data/raw/form/

Produces metadata manifest:
  data/raw/rvl_forms_manifest.csv
"""

import os
import sys
import csv
import io
import time
import tarfile
import requests
from PIL import Image

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "rvl_cdip_manifest.csv")

LABEL_MAP = {
    1: "form",
    2: "email",
    11: "invoice",
}

_BASE = "https://huggingface.co/datasets/rvl_cdip/resolve/main/data"
METADATA_URLS = {
    "train": f"{_BASE}/train.txt",
    "test":  f"{_BASE}/test.txt",
    "val":   f"{_BASE}/val.txt",
}
ARCHIVE_URL = f"{_BASE}/rvl-cdip.tar.gz"

IMAGES_PREFIX = "images/"


# ---------------------------------------------------------------------------
# Step 1 — Download and parse label metadata
# ---------------------------------------------------------------------------

def download_metadata():
    """Download the label files and return {image_path: (label_id, split)}."""
    path_to_meta = {}
    split_counts = {}

    for split_name, url in METADATA_URLS.items():
        print(f"  Downloading {split_name} labels …")
        r = requests.get(url)
        r.raise_for_status()

        lines = r.text.strip().splitlines()
        n_target = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rel_path, label_id_str = parts
            label_id = int(label_id_str)
            if label_id not in LABEL_MAP:
                continue
            full_path = IMAGES_PREFIX + rel_path
            path_to_meta[full_path] = (label_id, split_name)
            n_target += 1

        split_counts[split_name] = n_target
        print(f"    {split_name}: {n_target} target images "
              f"(out of {len(lines)} total)")

    print(f"\n  Total target images across all splits: {len(path_to_meta)}")
    return path_to_meta, split_counts


# ---------------------------------------------------------------------------
# Step 2 — Stream tar.gz and extract only matching images
# ---------------------------------------------------------------------------

def stream_and_extract(path_to_meta):
    """
    Stream the tar.gz archive over HTTP. For each member that matches a
    target path, read the TIFF bytes, convert to PNG, and save locally.
    The full archive is never stored on disk.
    """
    manifest_rows = []
    counts = {"form": 0, "email": 0, "invoice": 0}
    remaining = len(path_to_meta)

    print(f"\n  Opening archive stream ({ARCHIVE_URL}) …")
    print(f"  Looking for {remaining} target images inside ~38 GB archive.")
    print(f"  This will take a while — progress is printed every 50 saves.\n")

    resp = requests.get(ARCHIVE_URL, stream=True)
    resp.raise_for_status()

    raw_stream = resp.raw
    raw_stream.read = lambda n=-1, _read=raw_stream.read: _read(n)

    t0 = time.time()
    scanned = 0
    saved = 0

    try:
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                scanned += 1

                if not member.isfile():
                    continue

                name = member.name
                if name not in path_to_meta:
                    if scanned % 10000 == 0:
                        elapsed = time.time() - t0
                        pct = (saved / len(path_to_meta) * 100) if len(path_to_meta) > 0 else 0
                        print(f"  [scan] {scanned} files scanned, "
                              f"{saved}/{len(path_to_meta)} saved ({pct:.1f}%), "
                              f"{elapsed:.0f}s elapsed")
                    continue

                # --- This is a target image ---
                label_id, split_name = path_to_meta[name]
                class_name = LABEL_MAP[label_id]

                # Read image bytes from tar
                f_obj = tar.extractfile(member)
                if f_obj is None:
                    continue

                img_bytes = f_obj.read()

                # Convert TIFF to PNG and save
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                except Exception as e:
                    print(f"  WARN: could not open {name}: {e}")
                    continue

                counts[class_name] += 1
                idx = counts[class_name]
                fname = f"rvl_{split_name}_{idx:05d}.png"
                out_dir = os.path.join(RAW_DIR, class_name)
                os.makedirs(out_dir, exist_ok=True)
                abs_path = os.path.join(out_dir, fname)
                img.save(abs_path, format="PNG")

                manifest_rows.append({
                    "split": split_name,
                    "label_id": label_id,
                    "class_name": class_name,
                    "file_name": fname,
                    "local_path": os.path.join(class_name, fname),
                    "original_path": name,
                })

                saved += 1
                remaining -= 1

                if saved % 50 == 0:
                    elapsed = time.time() - t0
                    pct = saved / len(path_to_meta) * 100
                    print(f"  [save] {saved}/{len(path_to_meta)} images saved ({pct:.1f}%), "
                          f"form={counts['form']}, "
                          f"{elapsed:.0f}s elapsed")

                if remaining == 0:
                    print(f"\n  All target images found after scanning {scanned} files!")
                    break

    except Exception as e:
        print(f"\n  ERROR during archive streaming: {e}")
        import traceback
        traceback.print_exc()
        if saved > 0:
            print(f"  Partial results: {saved} images saved before error.")

    elapsed = time.time() - t0
    print(f"\n  Streaming complete: scanned {scanned} files in {elapsed:.0f}s")
    print(f"  Saved: form={counts['form']}, email={counts['email']}, invoice={counts['invoice']}, total={saved}")

    return manifest_rows, counts


# ---------------------------------------------------------------------------
# Step 3 — Write manifest
# ---------------------------------------------------------------------------

def write_manifest(rows):
    fieldnames = ["split", "label_id", "class_name", "file_name",
                  "local_path", "original_path"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Manifest saved: {MANIFEST_PATH} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for class_name in LABEL_MAP.values():
        os.makedirs(os.path.join(RAW_DIR, class_name), exist_ok=True)

    print("=" * 60)
    print("RVL-CDIP Selective Download: form (label=1) ONLY")
    print("=" * 60)

    # Step 1: Get metadata
    print("\n[Step 1] Downloading label metadata …")
    path_to_meta, split_counts = download_metadata()

    if not path_to_meta:
        print("ERROR: No target images found in metadata. Aborting.")
        sys.exit(1)

    # Step 2: Stream and extract
    print("\n[Step 2] Streaming archive and extracting target images …")
    manifest_rows, counts = stream_and_extract(path_to_meta)

    # Step 3: Write manifest
    print("\n[Step 3] Writing manifest …")
    write_manifest(manifest_rows)

    # Final report
    print("\n" + "=" * 60)
    print("FINAL COUNTS")
    print("=" * 60)
    for class_name in sorted(counts):
        class_dir = os.path.join(RAW_DIR, class_name)
        n_files = len([f for f in os.listdir(class_dir) if f.startswith("rvl_")])
        print(f"  {class_name}: {counts[class_name]} saved ({n_files} .png files on disk)")

    # Per-split breakdown
    split_class_counts = {}
    for row in manifest_rows:
        key = (row["split"], row["class_name"])
        split_class_counts[key] = split_class_counts.get(key, 0) + 1

    print("\nPer-split breakdown:")
    for (split, class_name), cnt in sorted(split_class_counts.items()):
        print(f"  {split:>10s} / {class_name:<10s}: {cnt}")

    print(f"\nTotal images saved: {sum(counts.values())}")


if __name__ == "__main__":
    main()
