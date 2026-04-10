# LaTeX Publication Files

This document provides information about the LaTeX source files and PDFs created for the three machine learning research projects.

## Overview

Professional LaTeX source files have been created for all three research projects, compiled into publication-ready PDFs suitable for academic journals and conferences.

## Files Created

### LaTeX Directory Structure

```
latex/
├── AI_Safety_RedTeam_Evaluation.tex      (18 KB) - LaTeX source
├── AI_Safety_RedTeam_Evaluation.pdf      (137 KB) - Compiled PDF
├── Breast_Cancer_Classification.tex      (23 KB) - LaTeX source
├── Breast_Cancer_Classification.pdf      (190 KB) - Compiled PDF
├── LLM_Bias_Detection.tex                (24 KB) - LaTeX source
├── LLM_Bias_Detection.pdf                (179 KB) - Compiled PDF
├── compile_latex.sh                      (3.6 KB) - Compilation script
└── README.md                             (8 KB) - Documentation
```

**Total: 8 files (6 documents + 1 script + 1 README)**

## Document Details

### 1. AI Safety Red-Team Evaluation
- **LaTeX Source**: `latex/AI_Safety_RedTeam_Evaluation.tex` (18 KB)
- **PDF Output**: `latex/AI_Safety_RedTeam_Evaluation.pdf` (137 KB)
- **Pages**: ~16 pages
- **Content**: Dual-stage framework for automated AI safety evaluation using LLM ensemble annotation and Bayesian ML classification

### 2. Breast Cancer Classification
- **LaTeX Source**: `latex/Breast_Cancer_Classification.tex` (23 KB)
- **PDF Output**: `latex/Breast_Cancer_Classification.pdf` (190 KB)
- **Pages**: ~19 pages
- **Content**: Ensemble methods for Wisconsin Breast Cancer classification with 99.12% accuracy

### 3. LLM Bias Detection
- **LaTeX Source**: `latex/LLM_Bias_Detection.tex` (24 KB)
- **PDF Output**: `latex/LLM_Bias_Detection.pdf` (179 KB)
- **Pages**: ~18 pages
- **Content**: Detecting publisher bias using LLM ensemble and Bayesian hierarchical methods

## LaTeX Features

All documents include:
- **Professional typography**: Times Roman font, 11pt
- **Academic structure**: Abstract, introduction, methodology, results, discussion, conclusions
- **Tables and figures**: Professional formatting with booktabs
- **Mathematical equations**: AMS math packages
- **References**: Embedded bibliography with numeric citations
- **Hyperlinks**: Clickable cross-references and URLs
- **PDF metadata**: Title, author, keywords embedded
- **Standards compliance**: IEEE 2830-2025, ISO/IEC 23894:2025

## Quick Start

### Compiling PDFs

```bash
cd latex

# Make script executable (first time only)
chmod +x compile_latex.sh

# Compile all documents
./compile_latex.sh

# Or compile individually
pdflatex AI_Safety_RedTeam_Evaluation.tex
pdflatex AI_Safety_RedTeam_Evaluation.tex  # Run twice for references
```

### Prerequisites

Install TeX Live:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# macOS
brew install --cask mactex
```

## Benefits of LaTeX Format

### For Academic Publishing
- **Industry Standard**: Required by most academic journals and conferences
- **arXiv Compatible**: Direct upload to arXiv preprint server
- **Journal Submission**: Easy conversion to journal-specific templates
- **Professional Quality**: Superior typography and mathematical typesetting

### For Reproducibility
- **Version Control**: Plain text format works with Git
- **Collaborative Editing**: Easy to track changes and collaborate
- **Automated Building**: CI/CD pipelines can auto-generate PDFs
- **Long-term Archival**: Plain text ensures future accessibility

### Comparison to Previous Format

| Feature | Markdown + Python | LaTeX |
|---------|------------------|-------|
| **Format** | Markdown → HTML → PDF (WeasyPrint) | LaTeX → PDF (pdflatex) |
| **File Size** | 96-140 KB | 137-190 KB |
| **Typography** | Web-based rendering | Professional TeX typesetting |
| **Math Support** | Limited | Excellent (AMS packages) |
| **Academic Standard** | Non-standard | Industry standard |
| **Journal Submission** | Requires conversion | Direct submission |
| **arXiv Compatibility** | PDF only | LaTeX source + PDF |

## Use Cases

### Academic Venues

1. **arXiv Preprints**
   - Upload LaTeX source (.tex files)
   - Auto-compilation on arXiv servers
   - Searchable, accessible format

2. **Conference Submissions**
   - Convert to conference templates (IEEE, ACL, NeurIPS)
   - Maintain all content structure
   - Easy adaptation to page limits

3. **Journal Submissions**
   - Submit LaTeX source to journals
   - Quick reformatting for journal styles
   - Track changes for revisions

### Professional Applications

1. **Research Portfolio**
   - Publication-quality documents
   - Professional presentation
   - Standards-compliant

2. **Industry Reports**
   - Technical documentation
   - Reproducible research
   - Version-controlled

3. **Grant Applications**
   - Preliminary results section
   - Technical appendices
   - Supporting documentation

## Documentation

Comprehensive documentation is available in `latex/README.md`:
- Detailed compilation instructions
- Customization guide
- Troubleshooting tips
- LaTeX package descriptions
- File format specifications

## Maintenance

### Updating Content

1. Edit `.tex` files with any text editor
2. Maintain LaTeX syntax
3. Recompile using `compile_latex.sh`
4. Verify PDF output

### Version Control

All LaTeX sources are version controlled:
- Plain text format for diffs
- Easy collaboration via Git
- Track changes over time
- Reproducible builds

## Standards Compliance

All documents comply with:
- **IEEE 2830-2025**: Transparent ML documentation
- **ISO/IEC 23894:2025**: AI Risk Management
- **Academic Standards**: Proper citation, reproducibility
- **Open Science**: Code/data availability statements

## Next Steps

### For Academic Publication

1. **arXiv Submission**
   - Upload LaTeX sources to arXiv
   - Obtain arXiv IDs
   - Update README with links

2. **Conference/Journal Submission**
   - Adapt to venue-specific templates
   - Submit LaTeX + PDF
   - Respond to reviewer feedback

3. **Citation Updates**
   - Update arXiv IDs once published
   - Add DOIs when available
   - Link from main README

### For Portfolio Enhancement

1. **GitHub Pages**
   - Host PDFs for easy access
   - Link from portfolio website
   - Update with publication info

2. **LinkedIn/CV**
   - Reference PDF publications
   - Include arXiv links
   - Highlight research impact

## Support

For questions or issues:
- **LaTeX Help**: See `latex/README.md` troubleshooting section
- **Project Issues**: https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/issues
- **Contact**: Derek Lankeaux via LinkedIn

## License

- **LaTeX Source**: MIT License
- **PDF Documents**: CC-BY-4.0
- **Code Examples**: MIT License

---

**Created**: April 10, 2026
**Author**: Derek Lankeaux, MS Applied Statistics
**Version**: 1.0.0
