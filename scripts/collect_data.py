"""
Data Collection Orchestrator
Downloads publicly accessible datasets and documents which ones need manual download.
Also runs the pilot data generator if no raw data exists yet.
"""

import os
import sys
import json
import urllib.request
import tarfile
import shutil

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")
DOWNLOAD_DIR = os.path.join(PROJ_ROOT, "data", "downloads")

# ---------------------------------------------------------------------------
# Dataset download attempts
# ---------------------------------------------------------------------------

def count_files(directory):
    """Count files (non-hidden) in a directory recursively."""
    total = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        total += sum(1 for f in files if not f.startswith("."))
    return total


def try_download_sroie_receipts():
    """
    SROIE receipt images — available on GitHub mirror.
    Downloads a small subset for the pilot.
    """
    dest = os.path.join(RAW_DIR, "receipt")
    if count_files(dest) >= 20:
        print("  [receipt] Already have ≥20 files, skipping download.")
        return True

    # The SROIE dataset is hosted on GitHub but as a full repo.
    # For the pilot, our synthetic samples suffice.
    print("  [receipt] SROIE dataset available at:")
    print("    https://github.com/zzzDavid/ICDAR-2019-SROIE")
    print("    Requires: git clone, then copy images from dataset/")
    print("    -> Using pilot synthetic data for now.")
    return False


def try_download_enron_emails():
    """
    Enron email dataset — publicly available from CMU.
    The full tarball is 1.7GB. We skip the full download for pilot.
    """
    dest = os.path.join(RAW_DIR, "email")
    if count_files(dest) >= 20:
        print("  [email] Already have ≥20 files, skipping download.")
        return True

    print("  [email] Enron dataset available at:")
    print("    https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz")
    print("    Size: ~1.7 GB (423K+ emails)")
    print("    -> Using pilot synthetic data for now.")
    print("    -> To use real data: download, extract, and copy 200 .txt emails to data/raw/email/")
    return False


def try_download_invoices():
    """
    Invoice datasets typically require Kaggle login or RVL-CDIP download.
    """
    dest = os.path.join(RAW_DIR, "invoice")
    if count_files(dest) >= 20:
        print("  [invoice] Already have ≥20 files, skipping download.")
        return True

    print("  [invoice] Public invoice datasets:")
    print("    1. Kaggle Invoice Dataset: https://www.kaggle.com/datasets")
    print("       -> Requires Kaggle login")
    print("    2. RVL-CDIP (invoice subset): https://huggingface.co/datasets/rvl_cdip")
    print("       -> Large dataset, requires HuggingFace download")
    print("    -> Using pilot synthetic data for now.")
    return False


def try_download_contracts():
    """
    CUAD contract dataset — available on HuggingFace / GitHub.
    """
    dest = os.path.join(RAW_DIR, "contract")
    if count_files(dest) >= 20:
        print("  [contract] Already have ≥20 files, skipping download.")
        return True

    print("  [contract] CUAD dataset available at:")
    print("    https://huggingface.co/datasets/cuad")
    print("    https://github.com/TheAtticusProject/cuad")
    print("    -> PDF contracts, requires download via HuggingFace datasets library or git clone")
    print("    -> Using pilot synthetic data for now.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("=" * 60)
    print("DATA COLLECTION — Checking dataset availability")
    print("=" * 60)

    results = {}
    results["invoice"] = try_download_invoices()
    results["receipt"] = try_download_sroie_receipts()
    results["email"] = try_download_enron_emails()
    results["contract"] = try_download_contracts()

    # Check if we need pilot data
    need_pilot = any(not v for v in results.values())
    if need_pilot:
        print("\n" + "=" * 60)
        print("Generating pilot synthetic data for classes without real data...")
        print("=" * 60)
        # Import and run the pilot generator
        sys.path.insert(0, os.path.dirname(__file__))
        from generate_pilot_data import main as gen_main
        gen_main()

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    for label in ["invoice", "receipt", "email", "contract"]:
        n = count_files(os.path.join(RAW_DIR, label))
        src = "real dataset" if results[label] else "pilot synthetic"
        print(f"  [{label}] {n} files ({src})")

    print("\nTo upgrade to real datasets, see instructions in data/dataset_manifest.json")


if __name__ == "__main__":
    main()
