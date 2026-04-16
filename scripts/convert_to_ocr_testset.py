"""
Convert PDF invoices to PNG for OCR pipeline testing.

Converts invoice PDFs to PNG images so the pipeline processes them
through Tesseract OCR instead of pdfplumber text extraction.
This tests the full pipeline as the professor would use it with
a photographed or scanned invoice.

Uses pypdfium2 (already installed as pdfplumber dependency) — no
poppler/ghostscript needed.

Usage:
    python scripts/convert_to_ocr_testset.py
    python scripts/convert_to_ocr_testset.py --input tests/invoices/ --output tests/invoices_ocr/ --degrade
"""

import argparse
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageFilter


def pdf_to_png(pdf_path: Path, png_path: Path,
               scale: float = 2.0, degrade: bool = False) -> None:
    """
    Render first page of a PDF to PNG.

    scale=2.0 → ~144 DPI, good for Tesseract.
    degrade=True → greyscale + slight blur, simulates a scanner.
    """
    doc    = pdfium.PdfDocument(str(pdf_path))
    page   = doc[0]
    bitmap = page.render(scale=scale)
    img    = bitmap.to_pil()

    if degrade:
        img = img.convert('L')                          # greyscale
        img = img.filter(ImageFilter.GaussianBlur(0.4)) # scanner softness
        img = img.filter(ImageFilter.SHARPEN)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(png_path), 'PNG')


def main():
    parser = argparse.ArgumentParser(
        description="Convert invoice PDFs to PNG for OCR testing"
    )
    parser.add_argument(
        "--input", default="tests/invoices",
        help="Directory of PDF invoices (default: tests/invoices/)"
    )
    parser.add_argument(
        "--output", default="tests/invoices_ocr",
        help="Output directory for PNG files (default: tests/invoices_ocr/)"
    )
    parser.add_argument(
        "--degrade", action="store_true",
        help="Apply greyscale + blur to simulate scanner degradation"
    )
    parser.add_argument(
        "--scale", type=float, default=2.0,
        help="Render scale factor (2.0 = ~144 DPI, default). Higher = sharper OCR."
    )
    args = parser.parse_args()

    proj_root  = Path(__file__).resolve().parent.parent
    input_dir  = proj_root / args.input
    output_dir = proj_root / args.output

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Converting {len(pdfs)} PDFs → PNG"
          f" ({'degraded' if args.degrade else 'clean'}, scale={args.scale})")
    print(f"Output: {output_dir}\n")

    for pdf_path in pdfs:
        png_path = output_dir / pdf_path.with_suffix('.png').name
        pdf_to_png(pdf_path, png_path, scale=args.scale, degrade=args.degrade)
        print(f"  {pdf_path.name} → {png_path.name}")

    print(f"\nDone. Add {output_dir.name}/ to your tests/ folder and run:")
    print(f"  python scripts/evaluate_pipeline.py --invoices {args.output}")


if __name__ == "__main__":
    main()
