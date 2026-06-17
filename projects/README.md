# Projects — Publication-Formatted Technical Reports

Two end-to-end data-science projects. Each is provided as a Markdown **source**
(`*_Report.md`) and a journal-style **publication PDF** (`*_Publication.pdf`)
rendered through the repository's publication pipeline.

| # | Project | Scope | Source | Publication |
|---|---------|-------|--------|-------------|
| 01 | AI Safety Red-Team Evaluation | LLM-ensemble harm annotation + Bayesian ML classification | [Report](./01_AI_Safety_RedTeam_Evaluation_Report.md) | [PDF](./01_AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| 02 | Bayesian Methods in Applied Classification | Two case studies: WBCD calibrated classifier + LLM-ensemble textbook bias detection | [Report](./02_Bayesian_Methods_in_Applied_Classification_Report.md) | [PDF](./02_Bayesian_Methods_in_Applied_Classification_Publication.pdf) |

Report 02 unifies the Wisconsin Diagnostic Breast Cancer case study and the
LLM-ensemble Textbook Bias Detection case study under a shared
Bayesian-methodology frame (priors, partial pooling, MCMC diagnostics,
calibration, decision policy).

## Regenerating

```bash
pip install -r ../requirements-pdf.txt
python ../professionalize_reports.py
```

`professionalize_reports.py` assembles each report's Markdown and renders the
publication PDF via `generate_publication_pdfs.generate_pdf()` — the same
journal-style format (title block, abstract/keywords styling, stripped TOC and
author sections, Dublin Core metadata) used for the rest of the portfolio.
Files follow the repo convention: `*_Report.md` for source, `*_Publication.pdf`
for the rendered publication.
