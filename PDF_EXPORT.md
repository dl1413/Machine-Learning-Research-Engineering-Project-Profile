# PDF Export Instructions

This document describes how to regenerate the four publication-ready PDFs from their LaTeX source files.

## Quick Start

```bash
chmod +x scripts/build_reports_pdf.sh
./scripts/build_reports_pdf.sh
```

Generated PDFs appear in **`pdf/out/`**.

## Prerequisites

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### macOS

```bash
brew install --cask mactex-no-gui
```

### Windows

Install [MiKTeX](https://miktex.org/download) and ensure `pdflatex` is on your PATH.

## Output Files

| PDF | LaTeX Source |
|-----|-------------|
| `pdf/out/AI_Safety_RedTeam_Report.pdf` | `latex/AI_Safety_RedTeam_Evaluation.tex` |
| `pdf/out/Breast_Cancer_Classification_Report.pdf` | `latex/Breast_Cancer_Classification.tex` |
| `pdf/out/LLM_Ensemble_Bias_Detection_Report.pdf` | `latex/LLM_Bias_Detection.tex` |
| `pdf/out/Machine_Learning_Research_Portfolio_2026.pdf` | `latex/Machine_Learning_Research_Portfolio_2026.tex` |

## Verify Prerequisites Only

```bash
./scripts/build_reports_pdf.sh --check
```

## Verbose / Debug Mode

```bash
VERBOSE=1 ./scripts/build_reports_pdf.sh
```

## Markdown Sources

The canonical Markdown sources (human-readable, version-controlled) are:

| Markdown | Description |
|----------|-------------|
| `AI Safety Red-Team Evaluation_ Technical Analysis Report.md` | AI Safety report |
| `Breast_Cancer_Classification_Report.md` | Breast Cancer report |
| `LLM_Ensemble_Bias_Detection_Report.md` | LLM Bias Detection report |
| `Machine_Learning_Research_Portfolio_2026.md` | Combined portfolio document |

The LaTeX files in `latex/` are the publication-formatted versions with academic typography, tables, references, and metadata embedded. They are maintained in sync with the Markdown sources but are independently editable for fine-grained formatting control.

## Document Specifications

All PDFs use:

- **Paper size:** A4, 1-inch margins
- **Font:** Times Roman, 11 pt body
- **Headers/Footers:** Project name (left), Author + Year (right), page number (centre)
- **Tables:** `booktabs` professional formatting
- **Cross-references:** Clickable hyperlinks via `hyperref`
- **PDF metadata:** Title, Author, Subject, Keywords embedded

## Standards Compliance

- **IEEE 2830-2025**: Transparent Machine Learning documentation
- **ISO/IEC 23894:2025**: AI Risk Management framework
- **Academic convention**: Standard LaTeX `article` class with embedded bibliography (no BibTeX required)
