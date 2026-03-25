from pathlib import Path
import os
import time

from google import genai

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "labeling"
CHUNK_DIRS = [
    RESULTS_DIR / "labeling_batches_recipient_name_chunks",
    RESULTS_DIR / "labeling_batches_invoice_number_chunks",
    RESULTS_DIR / "labeling_batches_total_amount_chunks",
]
RESPONSES_DIR = RESULTS_DIR / "label_responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

# Free-tier friendly choice for lightweight text labeling
MODEL = "gemini-3.1-flash-lite-preview"

PROMPT = """You are helping annotate candidate rows for invoice information extraction.

Task:
For each grouped case, assign labels to all candidates.

Rules:
- At most one candidate can be labeled 1 for a given doc_id and field.
- If none is clearly correct, all candidates must be 0.
- Be strict and conservative.
- Use the raw OCR text to understand the document before deciding.
- Do not guess from weak OCR if the evidence is poor.
- Do not infer missing text that is not actually present.

Field rules:
- recipient_name = billed recipient organization or person, not issuer, not remittance phrase, not address-only, not label text
- invoice_number = true invoice/reference/document number, not DATE/AMOUNT/NUMBER/years/obvious OCR junk
- total_amount = final total/amount due/balance due, not line items or unrelated amounts

Output only lines in this exact format:
doc_id=<doc_id> | field=<field> | candidate_text=<candidate_text> | label=<0 or 1>

Do not add explanations.
Do not skip any candidate.
Do not add markdown.
"""


def build_input_text(user_text: str) -> str:
    return f"{PROMPT}\n\n{user_text}"


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")

    client = genai.Client(api_key=api_key)

    for chunk_dir in CHUNK_DIRS:
        if not chunk_dir.exists():
            print(f"Skipping missing dir: {chunk_dir}")
            continue

        for chunk_file in sorted(chunk_dir.glob("*.txt")):
            out_file = RESPONSES_DIR / chunk_file.name

            if out_file.exists():
                print(f"Skipping existing response: {out_file.name}")
                continue

            user_text = chunk_file.read_text(encoding="utf-8")
            full_prompt = build_input_text(user_text)

            print(f"Labeling {chunk_file.name} ...")
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=full_prompt,
                )
                text = (response.text or "").strip()
                out_file.write_text(text, encoding="utf-8")
                print(f"Saved -> {out_file}")

                # Small pause to be nice to free-tier rate limits
                time.sleep(1.0)

            except Exception as e:
                print("\nStopped: Gemini request failed.")
                print(f"Chunk not processed: {chunk_file.name}")
                print(f"Error: {e}")
                return


if __name__ == "__main__":
    main()