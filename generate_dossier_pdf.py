#!/usr/bin/env python3
"""
Generate the Technical Capabilities Dossier PDF.

Reuses the publication pipeline in generate_publication_pdfs.py (academic
typography, metadata embedding, title block + abstract handling) to render
Technical_Capabilities_Dossier.md into an application-ready PDF.

Usage:
    python generate_dossier_pdf.py
"""

from pathlib import Path

from generate_publication_pdfs import generate_pdf

SOURCE = "Technical_Capabilities_Dossier.md"
OUTPUT = "Technical_Capabilities_Dossier.pdf"


def main():
    base = Path(__file__).parent
    md_path = base / SOURCE
    pdf_path = base / OUTPUT

    if not md_path.exists():
        raise SystemExit(f"Source not found: {md_path}")

    print("Generating Technical Capabilities Dossier PDF...\n")
    generate_pdf(str(md_path), str(pdf_path))

    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"    - File size: {size_kb:.1f} KB")
    print("\nDone.")


if __name__ == "__main__":
    main()
