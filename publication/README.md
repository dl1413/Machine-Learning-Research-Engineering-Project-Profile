# Publication Materials

This folder contains all materials related to generating publication-ready PDFs from the project's technical reports.

## Contents

- **`generate_publication_pdfs.py`** - Main Python script that converts markdown reports to publication-ready PDFs with proper formatting and metadata
- **`requirements-pdf.txt`** - Python dependencies required for PDF generation
- **`PDF_GENERATION_README.md`** - Comprehensive documentation on how to use the PDF generation pipeline
- **`PUBLICATION_READINESS_SUMMARY.md`** - Status summary of all projects' publication readiness

## Quick Start

```bash
# Install dependencies
pip install -r publication/requirements-pdf.txt

# Generate publication PDFs (run from repository root)
python publication/generate_publication_pdfs.py
```

The script will generate three publication-ready PDFs in the repository root:
- `AI_Safety_RedTeam_Evaluation_Publication.pdf`
- `Breast_Cancer_Classification_Publication.pdf`
- `LLM_Bias_Detection_Publication.pdf`

## Features

- LaTeX-style academic formatting with Times New Roman typography
- Comprehensive PDF metadata for academic indexing
- Dublin Core metadata for repository compatibility
- Optimized file sizes with PDF/1.7 compression
- Standards compliance (IEEE 2830-2025, ISO/IEC 23894:2025)

## Documentation

For detailed information about the PDF generation process, see [`PDF_GENERATION_README.md`](./PDF_GENERATION_README.md).

For publication readiness status and submission guidelines, see [`PUBLICATION_READINESS_SUMMARY.md`](./PUBLICATION_READINESS_SUMMARY.md).
