# Projects — Publication-Formatted Technical Reports

Two end-to-end data-science projects, published as Markdown source + rendered PDF.

| # | Project | Scope | Markdown | PDF |
|---|---------|-------|----------|-----|
| 01 | AI Safety Red-Team Evaluation | LLM-ensemble harm annotation + Bayesian ML classification | [.md](./01_AI_Safety_RedTeam_Evaluation.md) | [.pdf](./01_AI_Safety_RedTeam_Evaluation.pdf) |
| 02 | Bayesian Methods in Applied Classification | Two-case-study portfolio: WBCD calibrated classifier + LLM-ensemble textbook bias detection | [.md](./02_Bayesian_Methods_in_Applied_Classification.md) | [.pdf](./02_Bayesian_Methods_in_Applied_Classification.pdf) |

Report 02 unifies the Wisconsin Diagnostic Breast Cancer case study and the
LLM-ensemble Textbook Bias Detection case study under a shared
Bayesian-methodology frame (priors, partial pooling, MCMC diagnostics,
calibration, decision policy).

## Regenerating

```bash
pip install -r ../requirements-pdf.txt
python ../professionalize_reports.py
```

`professionalize_reports.py` builds report 01 by normalizing the AI-Safety
source report's metadata header, and builds report 02 by stitching the WBCD
and LLM-bias source reports together with a unified abstract, executive
summary, introduction, and synthesis/conclusion. Renders PDFs via the
existing WeasyPrint pipeline in `generate_publication_pdfs.py`.
