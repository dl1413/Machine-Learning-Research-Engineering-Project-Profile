# PDF Generation for 2026 Publication

This directory contains a Python script to generate publication-ready PDFs from markdown technical reports.

## 2026 Publication Optimizations

The `generate_publication_pdfs.py` script has been optimized for 2026 publication standards with:

### ✅ Enhanced PDF Metadata
- **Title, Author, Subject, Keywords** - Complete bibliographic information
- **Dublin Core metadata** - Academic database indexing support
- **Creation date and version tracking** - Publication provenance
- **Generator and producer fields** - Attribution and toolchain info

### ✅ File Size Optimization
- **PDF/1.7 format** - Modern compression algorithms
- **Image optimization** - Reduced file sizes without quality loss
- **Font subsetting** - Only embed used glyphs
- **Disabled unnecessary features** - Removed forms and attachments

### ✅ Academic Typography
- **Times New Roman** - Standard academic font family
- **Justified text** - Professional appearance with hyphenation
- **Proper margins** - 1-inch margins for print
- **Page numbering** - Bottom-center page numbers
- **Section formatting** - Clear hierarchical structure

### ✅ Standards Compliance
- **IEEE 2830-2025** - Transparent machine learning documentation
- **ISO/IEC 23894:2025** - AI risk management standards
- **Dublin Core** - Metadata for academic repositories
- **PDF/A hints** - Long-term archival compatibility

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements-pdf.txt
```

### Generate PDFs

```bash
# Run the generation script
python generate_publication_pdfs.py
```

The script will:
1. Read markdown source files
2. Extract metadata and keywords
3. Generate publication-formatted HTML
4. Convert to optimized PDF with full metadata
5. Report file sizes and completion status

## Output

The script generates three publication-ready PDFs:

1. **AI_Safety_RedTeam_Evaluation_Publication.pdf**
   - AI Safety Red-Team technical report
   - ~120 KB optimized file size

2. **Breast_Cancer_Classification_Publication.pdf**
   - Breast cancer ML classification report
   - ~110 KB optimized file size

3. **LLM_Bias_Detection_Publication.pdf**
   - LLM ensemble bias detection report
   - ~96 KB optimized file size

All PDFs include:
- Complete metadata for academic indexing
- Professional formatting for publication
- Optimized file sizes for distribution
- Embedded fonts for consistent rendering

## Metadata Fields

Each PDF contains comprehensive metadata:

```
Title:       [Extracted from markdown header]
Author:      Derek Lankeaux
Subject:     [Project description]
Keywords:    [First 10 keywords from document]
Creator:     WeasyPrint PDF Generator - 2026 Publication Pipeline
Producer:    Derek Lankeaux Machine Learning Research Portfolio
Date:        [Publication date from markdown]
Language:    English
Type:        Technical Report
Format:      application/pdf
```

## Requirements

- Python 3.8+
- WeasyPrint 60.0+ (PDF generation)
- Markdown 3.5+ (Markdown parsing)

See `requirements-pdf.txt` for complete dependency list.

## Technical Details

### PDF Features
- **Version**: PDF 1.7
- **Compression**: Enabled for all streams
- **Images**: Optimized automatically
- **Fonts**: Times New Roman (DejaVu Serif fallback)
- **Page size**: US Letter (8.5" × 11")
- **Margins**: 1" top/bottom, 0.85" left/right

### HTML Metadata
The script embeds comprehensive metadata in the HTML `<head>` section, which WeasyPrint preserves in the PDF:

- Standard meta tags (author, description, keywords, date)
- Dublin Core metadata for academic publishing
- Viewport settings for responsive rendering
- Language and encoding declarations

### Academic Formatting
- Abstract and keywords highlighted
- Section headings with consistent hierarchy
- Tables with professional styling
- Code blocks with monospace font
- Citations and references sections
- Page numbers (bottom-center, excluding first page)

## Troubleshooting

### Missing Dependencies
If you see `ModuleNotFoundError: No module named 'weasyprint'`:
```bash
pip install -r requirements-pdf.txt
```

### Font Rendering Issues
WeasyPrint uses system fonts. If Times New Roman is unavailable, it falls back to:
- DejaVu Serif (Linux)
- Georgia (fallback)

### File Size Concerns
The PDFs are optimized to balance quality and size:
- Images are automatically compressed
- Fonts are subsetted (only used glyphs embedded)
- PDF streams are compressed with modern algorithms

Typical sizes: 95-120 KB per 20-page technical report

## Maintenance

To update the PDFs after editing markdown sources:

1. Edit the markdown files (e.g., `AI Safety Red-Team Evaluation_ Technical Analysis Report (3).md`)
2. Run `python generate_publication_pdfs.py`
3. Verify the output PDFs

The script automatically:
- Removes "About the Author" sections
- Removes table of contents
- Cleans informal language
- Applies academic formatting
- Generates fresh metadata

## Version History

- **v2.0 (April 2026)** - Added 2026 publication optimizations
  - Comprehensive PDF metadata
  - Dublin Core academic metadata
  - Optimized file sizes
  - Enhanced error handling

- **v1.0 (April 2026)** - Initial publication pipeline
  - Basic PDF generation
  - Academic formatting
  - Metadata extraction
