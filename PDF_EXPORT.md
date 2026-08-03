# Publication PDF export

The repository ships four publication PDFs generated from the four root-level
technical reports. The canonical build uses `generate_publication_pdfs.py` and
writes the PDFs beside their source Markdown files.

## Quick start

Requires Python 3.11+ and the packages in `requirements-pdf.txt`:

```bash
python -m pip install -r requirements-pdf.txt
python generate_publication_pdfs.py
```

Run the lightweight repository check separately:

```bash
python scripts/validate_portfolio.py
```

## Generated artifacts

| Report | PDF |
|--------|-----|
| `AI Safety Red-Team Evaluation_ Technical Analysis Report.md` | `AI_Safety_RedTeam_Evaluation_Publication.pdf` |
| `Breast_Cancer_Classification_Report.md` | `Breast_Cancer_Classification_Publication.pdf` |
| `LLM_Ensemble_Bias_Detection_Report.md` | `LLM_Bias_Detection_Publication.pdf` |
| `RAG_Project_Report.md` | `RAG_Project_Publication.pdf` |

The committed PDFs are the portfolio deliverables. Rebuilding them is only
needed after changing a report or the publication renderer.

## Troubleshooting

- If `weasyprint` cannot import, reinstall `requirements-pdf.txt` in a clean
  virtual environment.
- If a report is skipped, confirm that its Markdown source is at the repository
  root and that the filename matches the `REPORTS` list in
  `generate_publication_pdfs.py`.
- CI runs `scripts/validate_portfolio.py` on every push to `main` and every pull
  request; it checks all four report/PDF pairs and local index links without
  requiring the PDF toolchain.
