# Projects — Publication-Formatted Technical Reports

Three end-to-end data-science projects. Each is provided as a Markdown **source**
(`*_Report.md`) and a journal-style **publication PDF** (`*_Publication.pdf`)
rendered through the repository's publication pipeline.

| # | Project | Scope | Source | Publication |
|---|---------|-------|--------|-------------|
| 01 | AI Safety Red-Team Evaluation | LLM-ensemble harm annotation + Bayesian ML classification | [Report](./01_AI_Safety_RedTeam_Evaluation_Report.md) | [PDF](./01_AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| 02 | LLM-Ensemble Textbook Bias Detection | LLM-as-judge ensemble + Bayesian hierarchical inference | [Report](./02_LLM_Ensemble_Bias_Detection_Report.md) | [PDF](./02_LLM_Ensemble_Bias_Detection_Publication.pdf) |
| 03 | Calibrated Binary Classification (WDBC) | Ensemble benchmarking, probability calibration, decision-policy tuning | [Report](./03_Breast_Cancer_Classification_Report.md) | [PDF](./03_Breast_Cancer_Classification_Publication.pdf) |

## Regenerating

```bash
pip install -r ../requirements-pdf.txt
python ../professionalize_reports.py
```

`professionalize_reports.py` normalizes each source report's metadata header to
one consistent template, then renders the publication PDF via
`generate_publication_pdfs.generate_pdf()` — the same journal-style format
(title block, abstract/keywords styling, stripped TOC and author sections,
Dublin Core metadata) used across the portfolio. Files follow the repo
convention: `*_Report.md` for source, `*_Publication.pdf` for the rendered
publication.
