#!/usr/bin/env bash
set -euo pipefail

echo "Building publication-ready PDFs..."

mkdir -p pdf/out

# Check for required dependencies
command -v pandoc >/dev/null 2>&1 || { echo "Pandoc is required but not installed. Install with: brew install pandoc"; exit 1; }
command -v xelatex >/dev/null 2>&1 || { echo "XeLaTeX is required but not installed. Install with: brew install --cask mactex-no-gui"; exit 1; }

echo "✓ Dependencies verified"

# PDF Build Configuration
PANDOC_ARGS="\
  --from gfm \
  --to pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V fontsize=11pt \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --standalone"

# Build individual reports
echo "Building AI Safety Red-Team Report..."
pandoc $PANDOC_ARGS "AI Safety Red-Team Evaluation_ Technical Analysis Report.md" \
  -o "pdf/out/01_AI_Safety_RedTeam_Report.pdf"
echo "✓ AI_Safety_RedTeam_Report.pdf"

echo "Building Breast Cancer Classification Report..."
pandoc $PANDOC_ARGS "Breast_Cancer_Classification_Report.md" \
  -o "pdf/out/02_Breast_Cancer_Classification_Report.pdf"
echo "✓ Breast_Cancer_Classification_Report.pdf"

echo "Building LLM Ensemble Bias Detection Report..."
pandoc $PANDOC_ARGS "LLM_Ensemble_Bias_Detection_Report.md" \
  -o "pdf/out/03_LLM_Ensemble_Bias_Detection_Report.pdf"
echo "✓ LLM_Ensemble_Bias_Detection_Report.pdf"

echo "Building RAG Project Report..."
pandoc $PANDOC_ARGS "RAG_Project_Report.md" \
  -o "pdf/out/04_RAG_Project_Report.pdf"
echo "✓ RAG_Project_Report.pdf"

echo ""
echo "========================================"
echo "All PDFs generated successfully!"
echo "========================================"
echo ""
echo "Output location: pdf/out/"
echo ""
ls -lh pdf/out/*.pdf
echo ""
echo "Total size: $(du -sh pdf/out | cut -f1)"
