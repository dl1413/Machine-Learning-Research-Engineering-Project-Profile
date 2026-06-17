# Projects — Final Publication Documents

Two final data-science projects, each delivered as a single journal-style
publication PDF rendered through the repository's publication pipeline.

| # | Project | Scope | Document |
|---|---------|-------|----------|
| 01 | AI Safety Red-Team Evaluation | LLM-ensemble harm annotation + Bayesian ML classification | [PDF](./01_AI_Safety_RedTeam_Evaluation.pdf) |
| 02 | Bayesian Methods in Applied Classification | Two case studies — WBCD calibrated classifier + LLM-ensemble textbook bias detection | [PDF](./02_Bayesian_Methods_in_Applied_Classification.pdf) |

Project 02 unifies the Wisconsin Diagnostic Breast Cancer case study and the
LLM-ensemble Textbook Bias Detection case study under a shared
Bayesian-methodology frame (priors, partial pooling, MCMC diagnostics,
calibration, decision policy), presented as Part A and Part B of one document.

## Regenerating

```bash
pip install -r ../requirements-pdf.txt
python ../professionalize_reports.py
```

`professionalize_reports.py` assembles each project's markdown in a temporary
build directory and renders the final PDF via
`generate_publication_pdfs.generate_pdf()` — the journal-style publication
format (title block, abstract/keywords styling, stripped TOC and author
sections, Dublin Core metadata) used across the portfolio. The deliverable
folder holds exactly one final document per project.
