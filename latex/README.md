# LaTeX Publication Documents

This directory contains LaTeX source files and compiled PDFs for the three machine learning research projects.

## Contents

### LaTeX Source Files (.tex)
1. **AI_Safety_RedTeam_Evaluation.tex** - AI Safety Red-Team Evaluation technical report
2. **Breast_Cancer_Classification.tex** - Breast Cancer ML Classification technical report
3. **LLM_Bias_Detection.tex** - LLM Ensemble Textbook Bias Detection technical report

### PDF Documents (.pdf)
1. **AI_Safety_RedTeam_Evaluation.pdf** (137 KB) - Compiled publication-ready PDF
2. **Breast_Cancer_Classification.pdf** (190 KB) - Compiled publication-ready PDF
3. **LLM_Bias_Detection.pdf** (179 KB) - Compiled publication-ready PDF

### Scripts
- **compile_latex.sh** - Bash script to compile LaTeX files to PDF

## Quick Start

### Prerequisites

Install TeX Live (LaTeX distribution):

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# macOS
brew install --cask mactex

# Windows
# Download and install MiKTeX from https://miktex.org/download
```

### Compiling PDFs

#### Option 1: Use the compilation script (recommended)

```bash
# Make script executable (first time only)
chmod +x compile_latex.sh

# Compile all LaTeX files
./compile_latex.sh

# Compile specific file(s)
./compile_latex.sh AI_Safety_RedTeam_Evaluation.tex
```

#### Option 2: Manual compilation

```bash
# Compile a single file
pdflatex AI_Safety_RedTeam_Evaluation.tex
pdflatex AI_Safety_RedTeam_Evaluation.tex  # Run twice for references

# Clean auxiliary files
rm *.aux *.log *.out *.toc
```

## Document Specifications

### LaTeX Features

All documents include:
- **Document Class**: article (11pt, A4 paper)
- **Font**: Times Roman (via `times` package)
- **Margins**: 1 inch on all sides
- **Headers/Footers**: Fancy headers with page numbers
- **Hyperlinks**: PDF hyperlinks via `hyperref` package
- **Tables**: Professional formatting with `booktabs`
- **Math**: AMS math packages for equations
- **Bibliography**: Embedded bibliography (no BibTeX required)
- **Metadata**: PDF metadata (title, author, keywords) embedded

### Page Layout
- **Page Size**: A4 (210mm × 297mm)
- **Font Size**: 11pt body text
- **Line Spacing**: 1.0 (single-spaced)
- **Margins**: 1.0 inch top/bottom, 1.0 inch left/right
- **Headers**: Project title (left) and author/date (right)
- **Footers**: Centered page numbers

### Document Structure

Each LaTeX document follows academic publication standards:

1. **Title Page**
   - Title
   - Author name and credentials
   - Institution affiliation
   - Date and version
   - AI standards compliance statement

2. **Abstract**
   - Concise summary (150-250 words)
   - Key findings and performance metrics
   - Statistical significance statements

3. **Keywords**
   - 10-15 relevant keywords for indexing

4. **Main Sections**
   - Introduction with motivation and objectives
   - Methodology and technical framework
   - Experimental results with tables and figures
   - Discussion and interpretation
   - Conclusions and future work

5. **Code and Data Availability**
   - GitHub repository links
   - Dataset sources with DOIs
   - Licensing information

6. **References**
   - Embedded bibliography
   - Numeric citation style
   - Complete bibliographic information

## PDF Features

### Metadata

All PDFs include comprehensive metadata:
- **Title**: Full project title
- **Author**: Derek Lankeaux
- **Subject**: Project description
- **Keywords**: Searchable keywords
- **PDF Version**: 1.5 (compatible with most readers)

### Accessibility
- Searchable text (not image-based)
- Clickable table of contents
- Hyperlinked cross-references
- Hyperlinked citations and URLs

### Quality
- **Resolution**: 600 DPI (print quality)
- **Color Space**: RGB (screen optimized)
- **Fonts**: Embedded for consistent rendering
- **Compression**: Optimized file size

## File Sizes

| Document | LaTeX Source | PDF Output |
|----------|--------------|------------|
| AI Safety Red-Team Evaluation | 14 KB | 137 KB |
| Breast Cancer Classification | 15 KB | 190 KB |
| LLM Bias Detection | 14 KB | 179 KB |
| **Total** | **43 KB** | **506 KB** |

## Customization

### Changing Document Format

To modify the document layout, edit the preamble in each `.tex` file:

```latex
% Change paper size to US Letter
\usepackage[margin=1in]{geometry}  % Change to letterpaper

% Adjust font size
\documentclass[12pt,a4paper]{article}  % Change 11pt to 12pt

% Modify margins
\usepackage[margin=0.75in]{geometry}  % Smaller margins
```

### Adding Content

To add sections or modify content:
1. Edit the `.tex` file with any text editor
2. Maintain LaTeX syntax (e.g., `\section{}`, `\subsection{}`)
3. Use `\cite{}` for citations (references are embedded)
4. Recompile to generate updated PDF

### Updating References

References are embedded in `\begin{thebibliography}` environment:

```latex
\bibitem{author2024}
Author, A. (2024).
\newblock Title of Paper.
\newblock \textit{Journal Name}, 10(2), 123-145.
```

## Troubleshooting

### Common Issues

**Problem**: `pdflatex: command not found`
**Solution**: Install TeX Live (see Prerequisites section)

**Problem**: Missing package errors (e.g., `! LaTeX Error: File 'times.sty' not found`)
**Solution**: Install additional LaTeX packages:
```bash
sudo apt-get install texlive-fonts-recommended texlive-latex-extra
```

**Problem**: PDF not generated after compilation
**Solution**: Check for errors in the log file:
```bash
pdflatex file.tex  # Look for errors in terminal output
cat file.log | grep "Error"  # Check log file for errors
```

**Problem**: Bibliography citations not working
**Solution**: All citations are embedded in the LaTeX files. No BibTeX compilation needed. Just run `pdflatex` twice.

**Problem**: Hyperlinks not working in PDF
**Solution**: Ensure `hyperref` package is loaded and compile twice

### Getting Help

For LaTeX-specific issues:
- LaTeX Documentation: https://www.latex-project.org/help/documentation/
- TeX Stack Exchange: https://tex.stackexchange.com/
- Overleaf Guides: https://www.overleaf.com/learn

For project-specific questions:
- Open an issue: https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/issues
- Contact: Derek Lankeaux via LinkedIn

## Compilation Script Details

The `compile_latex.sh` script provides:
- **Two-pass compilation**: Ensures references are resolved
- **Automatic cleanup**: Removes auxiliary files (.aux, .log, .out, .toc)
- **Error reporting**: Highlights compilation failures with color-coded output
- **File size reporting**: Shows PDF size after successful compilation
- **Batch processing**: Compiles all .tex files in directory if no arguments provided

### Script Usage Examples

```bash
# Compile all LaTeX files in directory
./compile_latex.sh

# Compile specific files
./compile_latex.sh AI_Safety_RedTeam_Evaluation.tex Breast_Cancer_Classification.tex

# Make script executable (first time)
chmod +x compile_latex.sh
```

## Standards Compliance

All documents comply with:
- **IEEE 2830-2025**: Transparent Machine Learning documentation standards
- **ISO/IEC 23894:2025**: AI Risk Management framework
- **Academic Publishing**: Standard LaTeX article class conventions
- **Reproducibility**: Fixed random seeds, version numbers, full methodology disclosure

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-10 | 1.0.0 | Initial LaTeX conversion from markdown sources |

## License

- **LaTeX Source Files**: MIT License
- **PDF Documents**: CC-BY-4.0 (Creative Commons Attribution)
- **Code Examples**: MIT License

## Author

**Derek Lankeaux, MS Applied Statistics**
Machine Learning Research Engineer
Rochester Institute of Technology

- GitHub: [@dl1413](https://github.com/dl1413)
- LinkedIn: [derek-lankeaux](https://linkedin.com/in/derek-lankeaux)
- Portfolio: [dl1413.github.io/LLM-Portfolio](https://dl1413.github.io/LLM-Portfolio)

---

**Last Updated**: April 10, 2026
**LaTeX Version**: TeX Live 2024
**PDF Version**: 1.5 (Acrobat 6.0 compatible)
