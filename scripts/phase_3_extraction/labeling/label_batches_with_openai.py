import os
from pathlib import Path
from openai import OpenAI, RateLimitError

PROJ_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJ_ROOT / "results" / "phase3_extraction" / "labeling"
CHUNK_DIRS = [
    RESULTS_DIR / "labeling_batches_recipient_name_chunks",
    RESULTS_DIR / "labeling_batches_invoice_number_chunks",
    RESULTS_DIR / "labeling_batches_total_amount_chunks",
]
RESPONSES_DIR = RESULTS_DIR / "label_responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5"
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

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

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

            print(f"Labeling {chunk_file.name} ...")
            try:
                response = client.responses.create(
                    model=MODEL,
                    input=[
                        {
                            "role": "developer",
                            "content": [{"type": "input_text", "text": PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_text}],
                        },
                    ],
                )
            except RateLimitError as e:
                print("\nStopped: API quota/billing is unavailable.")
                print("This usually means the API account has insufficient quota.")
                print(f"Chunk not processed: {chunk_file.name}")
                print(f"Error: {e}")
                break

            out_file.write_text(response.output_text, encoding="utf-8")
            print(f"Saved -> {out_file}")

if __name__ == "__main__":
    main()