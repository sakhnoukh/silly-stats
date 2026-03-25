from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "labeling"

INPUT_FILES = [
    "labeling_batches_recipient_name.txt",
    "labeling_batches_invoice_number.txt",
    "labeling_batches_total_amount.txt",
]

GROUP_SEPARATOR = "=" * 80
GROUPS_PER_CHUNK = {
    "labeling_batches_recipient_name.txt": 12,
    "labeling_batches_invoice_number.txt": 25,
    "labeling_batches_total_amount.txt": 25,
}


def split_groups(text):
    parts = text.split(GROUP_SEPARATOR)
    groups = []
    for part in parts:
        part = part.strip()
        if part:
            groups.append(GROUP_SEPARATOR + "\n" + part)
    return groups


def main():
    for fname in INPUT_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"Missing: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        groups = split_groups(text)
        chunk_size = GROUPS_PER_CHUNK.get(fname, 15)

        stem = fname.replace(".txt", "")
        out_dir = RESULTS_DIR / f"{stem}_chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for i in range(0, len(groups), chunk_size):
            chunk = groups[i:i + chunk_size]
            n += 1
            out_path = out_dir / f"{stem}_part_{n:02d}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(chunk))

        print(f"✓ {fname} -> {out_dir} ({n} chunks)")


if __name__ == "__main__":
    main()