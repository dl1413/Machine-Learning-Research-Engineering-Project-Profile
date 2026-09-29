# Derek Lankeaux

Data scientist and Applied Statistics M.S. candidate focused on rigorous
evaluation of machine-learning and LLM systems. I use experimental design,
Bayesian inference, and applied statistics to turn model results into
decision-relevant evidence.

[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/) · [Résumé](./Resume_Derek_Lankeaux.md)

[![Portfolio validation](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml/badge.svg)](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml)

## Portfolio

Explore four independent case studies in AI safety evaluation, applied
classification, LLM review, and retrieval-augmented generation. Each project
links to its full technical report and publication-formatted PDF, with evidence
and limitations made explicit.

| Project | Question | Methods | Evidence |
|---|---|---|---|
| **AI Safety Red-Team Evaluation** | How can LLM safety review scale without confusing model agreement with verified ground truth? | Ensemble annotation, supervised classification, Bayesian risk analysis | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) · [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) · [Project page](./project_packages/01_AI_Safety_RedTeam_Evaluation/) |
| **Breast Cancer Classification** | How do ensemble models compare across discrimination, calibration, and decision thresholds? | Benchmarking, calibration, feature selection, explainability | [Report](./Breast_Cancer_Classification_Report.md) · [PDF](./Breast_Cancer_Classification_Publication.pdf) · [Project page](./project_packages/02_Breast_Cancer_Classification/) |
| **LLM Ensemble Bias Detection** | How can multi-judge LLM review quantify disagreement and uncertainty at scale? | Rubric-based evaluation, reliability analysis, Bayesian hierarchical modeling | [Report](./LLM_Ensemble_Bias_Detection_Report.md) · [PDF](./LLM_Bias_Detection_Publication.pdf) · [Project page](./project_packages/03_LLM_Ensemble_Bias_Detection/) |
| **RAG Production Pipeline** | How should retrieval, grounding, confidence, and operational quality be evaluated together? | Hybrid retrieval, re-ranking, confidence calibration, observability design | [Report](./RAG_Project_Report.md) · [PDF](./RAG_Project_Publication.pdf) · [Project page](./project_packages/04_RAG_Production_Pipeline/) |

## What to review

- **Evaluation design:** Clear questions, defined evaluation setups, and methods matched to each problem.
- **Evidence quality:** Reliability, calibration, uncertainty, and validation are reported alongside headline metrics.
- **Operational relevance:** Results connect to review, triage, or monitoring decisions, with deployment considerations made explicit.
- **Responsible use:** Scope limits and further validation needs are stated for every project.

## Selected results

| Project | Reported result | Why it matters |
|---|---|---|
| AI Safety | 96.8% classification accuracy; Krippendorff's α = 0.81 | Separates annotation reliability from downstream classifier performance. |
| Breast Cancer | 99.12% held-out accuracy; ROC-AUC 0.9987 | Illustrates calibrated supervised-learning evaluation on the WDBC benchmark. |
| LLM Bias Detection | 67,500 ratings; Krippendorff's α = 0.84 | Demonstrates an uncertainty-aware workflow for large-scale content review. |
| RAG | 94.2% citation precision; 96.3% Recall@10 | Connects retrieval quality, grounding, and operational metrics in one systems design. |

All figures above are reported in the linked technical documents. The AI
Safety, LLM Bias Detection, and RAG results come from simulated evaluations
built to demonstrate each method; they are not measurements of real models,
publishers, or a deployed service. The Breast Cancer results use the public
WDBC benchmark and are not independent clinical validation.

## Technical focus

`Python` · `SQL` · `R` · `scikit-learn` · `XGBoost` · `LightGBM` · `PyMC` · `ArviZ` · `FastAPI` · `MLflow` · `SHAP` · `OpenAI` · `Anthropic` · `Qdrant` · `Docker` · `Kubernetes`

## Repository guide

```text
README.md                                      Portfolio overview
Resume_Derek_Lankeaux.md                       Résumé source
*_Report.md                                    Four technical reports
*_Publication.pdf                              Corresponding publication PDFs
project_packages/                              Per-project reader guides
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
