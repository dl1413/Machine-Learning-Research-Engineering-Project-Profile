# Derek Lankeaux

Data Scientist and Applied Statistics M.S. candidate focused on experimental
design, Bayesian inference, machine learning, and LLM evaluation.

[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/) · [Résumé](./Resume_Derek_Lankeaux.md)

[![Portfolio validation](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml/badge.svg)](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml)

## Portfolio

This repository contains four independent technical case studies. Each project
links to a full Markdown report and a publication-formatted PDF; the summary
below is intended to make the evidence and the scope easy to evaluate.

| Project | Question | Methods | Evidence |
|---|---|---|---|
| **AI Safety Red-Team Evaluation** | How can safety evaluation scale beyond manual review? | LLM ensemble annotation, supervised classification, Bayesian risk analysis | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) · [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) · [Project page](./project_packages/01_AI_Safety_RedTeam_Evaluation/) |
| **Breast Cancer Classification** | Which ensemble methods perform well on WDBC diagnostic features? | Benchmarking, calibration, feature selection, explainability | [Report](./Breast_Cancer_Classification_Report.md) · [PDF](./Breast_Cancer_Classification_Publication.pdf) · [Project page](./project_packages/02_Breast_Cancer_Classification/) |
| **LLM Ensemble Bias Detection** | Can multiple LLM judges support uncertainty-aware content review? | Rubric-based LLM evaluation, reliability analysis, Bayesian hierarchical modeling | [Report](./LLM_Ensemble_Bias_Detection_Report.md) · [PDF](./LLM_Bias_Detection_Publication.pdf) · [Project page](./project_packages/03_LLM_Ensemble_Bias_Detection/) |
| **RAG Production Pipeline** | How can retrieval, grounding, and monitoring improve RAG system design? | Hybrid retrieval, re-ranking, confidence calibration, observability design | [Report](./RAG_Project_Report.md) · [PDF](./RAG_Project_Publication.pdf) · [Project page](./project_packages/04_RAG_Production_Pipeline/) |

## What to review

- **Problem framing and evaluation design:** Each report documents a defined problem, data/evaluation setup, and methodological choices.
- **Statistical rigor:** The projects use cross-validation, inter-rater reliability, confidence intervals, Bayesian inference, or calibration as appropriate to the task.
- **Decision relevance:** The work connects model results to practical review, triage, or monitoring decisions rather than treating a headline metric as sufficient on its own.
- **Responsible use:** Each package page states the limits of the project and the validation needed before any real-world use.

## Selected results

| Project | Reported result | Why it matters |
|---|---|---|
| AI Safety | 96.8% classification accuracy; Krippendorff's α = 0.81 | Separates annotation reliability from downstream classifier performance. |
| Breast Cancer | 99.12% held-out accuracy; ROC-AUC 0.9987 | Illustrates calibrated supervised-learning evaluation on the WDBC benchmark. |
| LLM Bias Detection | 67,500 ratings; Krippendorff's α = 0.84 | Demonstrates an uncertainty-aware workflow for large-scale content review. |
| RAG | 94.2% citation precision; 96.3% Recall@10 | Connects retrieval quality, grounding, and operational metrics in one systems design. |

All figures above are reported in the linked technical documents. They are
project-evaluation results, not independent clinical validation, product
performance guarantees, or evidence of a live service.

## Technical focus

`Python` · `SQL` · `R` · `scikit-learn` · `XGBoost` · `LightGBM` · `PyMC` · `ArviZ` · `FastAPI` · `MLflow` · `SHAP` · `OpenAI` · `Anthropic` · `Qdrant` · `Docker` · `Kubernetes`

## Repository guide

```text
README.md                                      Portfolio overview
Resume_Derek_Lankeaux.md                       Résumé source
*_Report.md                                    Four technical reports
*_Publication.pdf                              Corresponding publication PDFs
project_packages/                              Per-project reader guides and PDFs
generate_publication_pdfs.py                   Canonical PDF generator
requirements-pdf.txt                           PDF-generation dependencies
PDF_EXPORT.md                                  Build and validation instructions
scripts/validate_portfolio.py                  Artifact and local-link validation
.github/workflows/validate-portfolio.yml       GitHub Actions quality check
```

For PDF regeneration and validation, see [PDF export instructions](./PDF_EXPORT.md).
The source notebooks referenced in the reports are not distributed in this
repository; the reports and PDFs are the public portfolio artifacts.

## Contact

Open to 2026 data science, applied ML, and LLM-evaluation opportunities. The
best way to connect is on [LinkedIn](https://linkedin.com/in/derek-lankeaux).
