# Derek Lankeaux

Machine learning engineer focused on model evaluation, error analysis, and
diagnostic ML, with graduate training in applied statistics at RIT.

[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Résumé](./Resume_Derek_Lankeaux.md)

[![Portfolio validation](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml/badge.svg)](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile/actions/workflows/validate-portfolio.yml)

## Portfolio

Five independent technical case studies. Two use real data; three are
methodology case studies built on simulated data to work out an evaluation
design before running it for real. Each project page states which, along with
the limits of the work.

| Project | Question | Data | Evidence |
|---|---|---|---|
| **Breast Cancer Classification** | Which ensemble methods perform well on WDBC diagnostic features, and where do they fail? | Real (WDBC benchmark) | [Report](./Breast_Cancer_Classification_Report.md) · [PDF](./Breast_Cancer_Classification_Publication.pdf) · [Project page](./project_packages/02_Breast_Cancer_Classification/) |
| **Clinical Privacy vs. Predictive Utility** | How much model performance does each de-identification strategy cost on hospital data? | Real (Diabetes 130-US, 101,766 encounters) | [Project page](./project_packages/05_Clinical_Privacy_Utility/) |
| **AI Safety Red-Team Evaluation** | How can safety evaluation scale beyond manual review? | Simulated | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) · [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) · [Project page](./project_packages/01_AI_Safety_RedTeam_Evaluation/) |
| **LLM Ensemble Bias Detection** | Can multiple LLM judges support uncertainty-aware content review? | Simulated | [Report](./LLM_Ensemble_Bias_Detection_Report.md) · [PDF](./LLM_Bias_Detection_Publication.pdf) · [Project page](./project_packages/03_LLM_Ensemble_Bias_Detection/) |
| **RAG Production Pipeline** | How can retrieval, grounding, and monitoring improve RAG system design? | Simulated | [Report](./RAG_Project_Report.md) · [PDF](./RAG_Project_Publication.pdf) · [Project page](./project_packages/04_RAG_Production_Pipeline/) |

## Selected results

| Project | Result | Why it matters |
|---|---|---|
| Breast Cancer | 99.12% held-out accuracy (113/114); all 43 malignant test cases detected; the single error was a benign tumor flagged as malignant | Error analysis at the level of individual cases, not just an aggregate metric. |
| Clinical Privacy | Generalization halved unique-record re-identification risk (15.4% → 7.6%) for a 0.3-point AUC cost; AUC fell to 0.515 for patients aged 90+ | Quantifies a privacy–utility trade-off and surfaces a subgroup failure the overall AUC (0.672) hides. |
| AI Safety (simulated) | Krippendorff's α = 0.81; 96.8% accuracy, 95.6% recall against ensemble labels | Separates annotation reliability from downstream classifier performance. |
| LLM Bias Detection (simulated) | 67,500 ratings; Krippendorff's α = 0.84 | An uncertainty-aware workflow for large-scale content review. |
| RAG (simulated) | 94.2% citation precision; 95.4% average Recall@10 | Connects retrieval quality, grounding, and latency in one systems design. |

The Breast Cancer results use the public WDBC benchmark and are not clinical
validation. The Clinical Privacy results use the public Diabetes 130-US
Hospitals dataset. The AI Safety, LLM Bias Detection, and RAG results come
from simulated evaluations; they are not measurements of real models,
publishers, or a deployed service.

## What to review

- **Error and failure analysis:** Each project looks past the headline metric to individual errors, subgroups, thresholds, or failure modes.
- **Evaluation design:** Train/test separation, cross-validation, inter-rater reliability, calibration, and confidence intervals where they apply.
- **Honest scope:** Each project page states what the evidence does and does not support.

## Technical focus

`Python` · `PyTorch` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `PyMC` · `ArviZ` · `SQL` · `R` · `FastAPI` · `MLflow` · `Docker` · `Kubernetes` · `OpenAI` · `Anthropic` · `Qdrant`

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

Open to machine learning engineering, applied ML, and model-evaluation roles,
including medical imaging. The best way to connect is on
[LinkedIn](https://linkedin.com/in/derek-lankeaux).
