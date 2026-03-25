"""
Create a manifest CSV for partially downloaded RVL-CDIP form images.

This allows us to proceed with Phase 1 while the full download continues.
"""

import os
import csv

PROJ_ROOT = "/Users/sofiaclaudiabonoan/Desktop/silly-stats"
RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")
FORM_DIR = os.path.join(RAW_DIR, "form")
MANIFEST_PATH = os.path.join(RAW_DIR, "rvl_forms_manifest.csv")

# Get all downloaded form images
form_images = sorted([f for f in os.listdir(FORM_DIR) if f.endswith(".png")])

print(f"Found {len(form_images)} form images in {FORM_DIR}")

# Create manifest
with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label_id", "split"])
    for i, img in enumerate(form_images):
        # Assign to train/val/test based on index
        if i % 100 < 70:
            split = "train"
        elif i % 100 < 85:
            split = "val"
        else:
            split = "test"
        writer.writerow([f"form/{img}", 1, split])

print(f"✓ Manifest created: {MANIFEST_PATH}")
print(f"  Rows: {len(form_images)}")
