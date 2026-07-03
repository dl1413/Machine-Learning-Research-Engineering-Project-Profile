# Projects — Final Publication Documents

Four final data-science projects, each delivered as a single journal-style
publication PDF rendered through the repository's publication pipeline.

| # | Project | Scope | Document |
|---|---------|-------|----------|
| 01 | AI Safety Red-Team Evaluation | LLM-ensemble harm annotation + Bayesian ML classification | [PDF](./01_AI_Safety_RedTeam_Evaluation.pdf) |
| 02 | Breast Cancer Classification | Ensemble-learning classification + calibration for diagnostic support | [PDF](./02_Breast_Cancer_Classification.pdf) |
| 03 | LLM Ensemble Bias Detection | Bayesian hierarchical publisher-level bias detection | [PDF](./03_LLM_Ensemble_Bias_Detection.pdf) |
| 04 | RAG System Engineering | Retrieval-augmented generation with grounding, guardrails, and evaluation | [PDF](./04_RAG_System_Engineering.pdf) |

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
