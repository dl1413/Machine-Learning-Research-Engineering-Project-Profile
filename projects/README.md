# Projects — Professionalized Technical Reports

Publication-formatted versions of three end-to-end data science projects. Each report
shares a consistent title block (project, author, role, institution, version, standards
compliance) and is provided as both Markdown source and a rendered PDF.

| # | Project | Domain | Markdown | PDF |
|---|---------|--------|----------|-----|
| 01 | AI Safety Red-Team Evaluation | GenAI evaluation · LLM ensemble · Bayesian risk modeling | [.md](./01_AI_Safety_RedTeam_Evaluation.md) | [.pdf](./01_AI_Safety_RedTeam_Evaluation.pdf) |
| 02 | LLM Ensemble Textbook Bias Detection | LLM-as-judge · Bayesian hierarchical inference | [.md](./02_LLM_Ensemble_Bias_Detection.md) | [.pdf](./02_LLM_Ensemble_Bias_Detection.pdf) |
| 03 | Breast Cancer Classification | Calibrated predictive modeling · decision-policy tuning | [.md](./03_Breast_Cancer_Classification.md) | [.pdf](./03_Breast_Cancer_Classification.pdf) |

## Regenerating

```bash
pip install -r ../requirements-pdf.txt
python ../professionalize_reports.py
```

`professionalize_reports.py` normalizes each source report's header to the shared
template and renders the PDF via the WeasyPrint pipeline in `generate_publication_pdfs.py`.
