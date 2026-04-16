#!/bin/bash
#
# LaTeX to PDF Compilation Script
# Generates publication-ready PDFs from LaTeX source files
#
# Usage: ./compile_latex.sh [file1.tex file2.tex ...] or ./compile_latex.sh (compiles all)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "LaTeX to PDF Compilation Script"
echo "======================================================================"
echo ""

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}Error: pdflatex not found. Please install TeX Live or MiKTeX.${NC}"
    echo "On Ubuntu/Debian: sudo apt-get install texlive-latex-extra texlive-fonts-recommended"
    echo "On macOS: brew install --cask mactex"
    exit 1
fi

# Function to compile a single LaTeX file
compile_tex() {
    local texfile="$1"
    local basename="${texfile%.tex}"

    echo -e "${YELLOW}Compiling: $texfile${NC}"

    # First pass
    echo "  Running pdflatex (pass 1/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error "$texfile" > /dev/null 2>&1

    # Second pass (for references, TOC, etc.)
    echo "  Running pdflatex (pass 2/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error "$texfile" > /dev/null 2>&1

    # Clean up auxiliary files
    echo "  Cleaning auxiliary files..."
    rm -f "${basename}.aux" "${basename}.log" "${basename}.out" "${basename}.toc" \
          "${basename}.lof" "${basename}.lot" "${basename}.bbl" "${basename}.blg" \
          "${basename}.nav" "${basename}.snm" "${basename}.vrb"

    # Check if PDF was created
    if [ -f "${basename}.pdf" ]; then
        local filesize=$(du -h "${basename}.pdf" | cut -f1)
        echo -e "${GREEN}✓ Success: ${basename}.pdf (${filesize})${NC}"
        return 0
    else
        echo -e "${RED}✗ Error: Failed to generate ${basename}.pdf${NC}"
        return 1
    fi
}

# Determine which files to compile
if [ $# -eq 0 ]; then
    # No arguments: compile all .tex files in current directory
    TEX_FILES=(*.tex)
    if [ ! -e "${TEX_FILES[0]}" ]; then
        echo -e "${RED}Error: No .tex files found in current directory${NC}"
        exit 1
    fi
else
    # Arguments provided: compile specified files
    TEX_FILES=("$@")
fi

# Track success/failure
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=${#TEX_FILES[@]}

echo "Files to compile: ${TOTAL_COUNT}"
echo ""

# Compile each file
for texfile in "${TEX_FILES[@]}"; do
    if [ ! -f "$texfile" ]; then
        echo -e "${RED}Warning: $texfile not found, skipping${NC}"
        ((FAIL_COUNT++))
        continue
    fi

    if compile_tex "$texfile"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Summary
echo "======================================================================"
echo "Compilation Summary"
echo "======================================================================"
echo "Total files:    $TOTAL_COUNT"
echo -e "${GREEN}Successful:     $SUCCESS_COUNT${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed:         $FAIL_COUNT${NC}"
else
    echo "Failed:         $FAIL_COUNT"
fi
echo "======================================================================"

# Exit with appropriate code
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All PDFs generated successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some compilations failed. Check error messages above.${NC}"
    exit 1
fi
