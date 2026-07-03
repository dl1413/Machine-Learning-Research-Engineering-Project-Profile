#!/usr/bin/env bash
#
# build_reports_pdf.sh
# Generates publication-ready PDFs from LaTeX source files.
#
# Usage:
#   ./scripts/build_reports_pdf.sh           # build all 4 PDFs
#   ./scripts/build_reports_pdf.sh --check   # verify prerequisites only
#
# Output: pdf/out/
#   - AI_Safety_RedTeam_Report.pdf
#   - Breast_Cancer_Classification_Report.pdf
#   - LLM_Ensemble_Bias_Detection_Report.pdf
#   - Machine_Learning_Research_Portfolio_2026.pdf

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEX_DIR="${REPO_ROOT}/latex"
OUT_DIR="${REPO_ROOT}/pdf/out"

# ── Report definitions ────────────────────────────────────────────────────────
# Format: "source.tex|output.pdf"
REPORTS=(
    "AI_Safety_RedTeam_Evaluation.tex|AI_Safety_RedTeam_Report.pdf"
    "Breast_Cancer_Classification.tex|Breast_Cancer_Classification_Report.pdf"
    "LLM_Bias_Detection.tex|LLM_Ensemble_Bias_Detection_Report.pdf"
    "Machine_Learning_Research_Portfolio_2026.tex|Machine_Learning_Research_Portfolio_2026.pdf"
)

# ── Prerequisites check ───────────────────────────────────────────────────────
check_prerequisites() {
    local missing=0
    if ! command -v pdflatex &>/dev/null; then
        echo -e "${RED}Error: pdflatex not found.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-extra texlive-fonts-recommended"
        echo "  macOS:         brew install --cask mactex-no-gui"
        missing=1
    fi
    if [ ! -d "${LATEX_DIR}" ]; then
        echo -e "${RED}Error: latex/ source directory not found at ${LATEX_DIR}${NC}"
        missing=1
    fi
    return $missing
}

# ── Compile a single LaTeX file ───────────────────────────────────────────────
compile_tex() {
    local src_tex="${LATEX_DIR}/$1"
    local out_pdf="${OUT_DIR}/$2"
    local base="${src_tex%.tex}"

    echo -e "${YELLOW}  Compiling: $1${NC}"

    if [ ! -f "${src_tex}" ]; then
        echo -e "${RED}    ✗ Source not found: ${src_tex}${NC}"
        return 1
    fi

    # Two-pass compile (resolves TOC, cross-refs, etc.)
    pdflatex -interaction=nonstopmode -halt-on-error \
             -output-directory="${LATEX_DIR}" "${src_tex}" >/dev/null 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error \
             -output-directory="${LATEX_DIR}" "${src_tex}" >/dev/null 2>&1

    local compiled_pdf="${base}.pdf"

    if [ ! -f "${compiled_pdf}" ]; then
        echo -e "${RED}    ✗ Compilation failed — re-run with VERBOSE=1 for details${NC}"
        return 1
    fi

    # Move to output directory with canonical name
    mv "${compiled_pdf}" "${out_pdf}"

    # Cleanup auxiliary files
    rm -f "${base}.aux" "${base}.log" "${base}.out" "${base}.toc" \
          "${base}.lof" "${base}.lot" "${base}.bbl" "${base}.blg"

    local size
    size=$(du -h "${out_pdf}" | cut -f1)
    echo -e "${GREEN}    ✓ ${2} (${size})${NC}"
    return 0
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo "======================================================================"
    echo "  ML Research Portfolio — PDF Build Pipeline"
    echo "======================================================================"
    echo ""

    # Prerequisites
    if ! check_prerequisites; then
        exit 1
    fi

    # Handle --check flag
    if [[ "${1:-}" == "--check" ]]; then
        echo -e "${GREEN}All prerequisites satisfied.${NC}"
        exit 0
    fi

    # Create output directory
    mkdir -p "${OUT_DIR}"

    # Compile all reports
    local success=0 fail=0 total=${#REPORTS[@]}
    for entry in "${REPORTS[@]}"; do
        local tex="${entry%%|*}"
        local pdf="${entry##*|}"
        if compile_tex "${tex}" "${pdf}"; then
            success=$((success + 1))
        else
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo "======================================================================"
    echo "  Build Summary: ${success}/${total} PDFs generated successfully"
    echo "  Output directory: ${OUT_DIR}"
    echo "======================================================================"

    if [ "${fail}" -gt 0 ]; then
        echo -e "${YELLOW}  ${fail} build(s) failed. Set VERBOSE=1 and re-run for full LaTeX output.${NC}"
        exit 1
    fi

    echo -e "${GREEN}  All PDFs ready.${NC}"
}

# Verbose mode: show full pdflatex output when VERBOSE=1
if [[ "${VERBOSE:-0}" == "1" ]]; then
    compile_tex() {
        local src_tex="${LATEX_DIR}/$1"
        local out_pdf="${OUT_DIR}/$2"
        local base="${src_tex%.tex}"
        echo -e "${YELLOW}  Compiling (verbose): $1${NC}"
        pdflatex -interaction=nonstopmode -output-directory="${LATEX_DIR}" "${src_tex}" || true
        pdflatex -interaction=nonstopmode -output-directory="${LATEX_DIR}" "${src_tex}"
        [ -f "${base}.pdf" ] && mv "${base}.pdf" "${out_pdf}"
        rm -f "${base}.aux" "${base}.log" "${base}.out" "${base}.toc"
        echo -e "${GREEN}    ✓ ${2}${NC}"
    }
fi

main "$@"
