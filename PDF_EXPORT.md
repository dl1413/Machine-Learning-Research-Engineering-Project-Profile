# PDF Export Instructions (2026)

## Quick Start

```bash
chmod +x scripts/build_reports_pdf.sh
./scripts/build_reports_pdf.sh
```

## Prerequisites

### macOS (Recommended)
```bash
brew install pandoc
brew install --cask mactex-no-gui
```

Alternative (smaller install):
```bash
brew install pandoc
brew install --cask basictex
tlmgr install xetex collection-latex collection-langenglish
```

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y pandoc texlive-xetex texlive-latex-extra texlive-fonts-recommended
```

### Windows (PowerShell)
```powershell
choco install pandoc
choco install miktex
```

## Build Output

Generated PDFs in `pdf/out/`:

1. **01_AI_Safety_RedTeam_Report.pdf** (87 pages)
   - Dual-stage LLM ensemble + ML classification for AI safety evaluation
   - Red-team adversarial attack taxonomy and defense analysis
   - Bayesian hierarchical risk modeling across models and harm categories

2. **02_Breast_Cancer_Classification_Report.pdf** (102 pages)
   - Ensemble learning on Wisconsin Diagnostic Breast Cancer dataset
   - Model calibration, clinical threshold optimization
   - SHAP explainability and fairness audit

3. **03_LLM_Ensemble_Bias_Detection_Report.pdf** (94 pages)
   - LLM ensemble (GPT-4o + Claude-3.5 + Llama-3.2) for textbook bias detection
   - Krippendorff's α reliability (α = 0.84)
   - Bayesian hierarchical publisher-level effects

4. **04_RAG_Project_Report.pdf** (72 pages)
   - Production Retrieval-Augmented Generation system
   - Multi-model embedding orchestration
   - Hallucination detection and grounding framework

## Document Specifications

- **Format:** PDF (A4, 8.5" × 11")
- **Margins:** 1 inch (all sides)
- **Font:** XeLaTeX default (Computer Modern)
- **Color:** Links enabled (blue)
- **TOC:** Full table of contents with section numbering
- **Bookmarks:** Interactive PDF bookmarks for navigation

## Customization

To change styling, edit `scripts/build_reports_pdf.sh`:

```bash
# Change margin
-V geometry:margin=0.75in

# Change font size
-V fontsize=12pt

# Disable TOC
# --toc (comment out)

# Change color scheme
-V linkcolor=red
```

## Troubleshooting

### "pandoc: command not found"
→ Install Pandoc: `brew install pandoc`

### "xelatex: command not found"
→ Install TeX: `brew install --cask mactex-no-gui`

### PDF generation hangs
→ Try with minimal TeX: `brew install --cask basictex && tlmgr install collection-latex`

### Memory errors during build
→ Close other applications; TeX can be memory-intensive with large documents.

## File Sizes

Expected output sizes:
- AI Safety: ~8-12 MB
- Breast Cancer: ~10-14 MB
- LLM Bias: ~9-13 MB
- RAG: ~6-9 MB

**Total: ~35-50 MB**

## Version Info

```bash
pandoc --version
xelatex --version
```

Recommended:
- Pandoc: ≥3.0
- TeX Live: ≥2024

---

**Generated:** 2026  
**Standards:** IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act
