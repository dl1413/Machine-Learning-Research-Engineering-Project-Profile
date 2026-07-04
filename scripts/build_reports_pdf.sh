#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/pdf/out"
TEMPLATE="${ROOT_DIR}/pdf/ieee-report-template.tex"

mkdir -p "${OUT_DIR}"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc not found in PATH." >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "Error: xelatex not found in PATH." >&2
  exit 1
fi

build_one() {
  local input_md="$1"
  local output_pdf="$2"

  pandoc "${ROOT_DIR}/${input_md}" \
    --from markdown+tex_math_dollars+fenced_code_attributes \
    --to pdf \
    --pdf-engine=xelatex \
    --template="${TEMPLATE}" \
    --number-sections \
    --toc \
    --toc-depth=3 \
    --no-highlight \
    --variable geometry:margin=1in \
    --variable colorlinks=true \
    --output "${OUT_DIR}/${output_pdf}"
}

echo "Building publication-ready PDFs with Pandoc + XeLaTeX..."

build_one "AI Safety Red-Team Evaluation_ Technical Analysis Report.md" "AI_Safety_RedTeam_Report.pdf"
build_one "Breast_Cancer_Classification_Report.md" "Breast_Cancer_Classification_Report.pdf"
build_one "LLM_Ensemble_Bias_Detection_Report.md" "LLM_Ensemble_Bias_Detection_Report.pdf"
build_one "RAG_Project_Report.md" "RAG_Project_Report.pdf"

echo "Done. Output files:"
ls -1 "${OUT_DIR}"/*.pdf
