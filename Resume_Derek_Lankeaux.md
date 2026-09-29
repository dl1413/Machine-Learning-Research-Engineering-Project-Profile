# Derek Lankeaux

**Machine Learning Engineer | Model Evaluation & Applied ML**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile)

## Summary

Machine learning engineer with graduate training in applied statistics,
focused on model evaluation, error analysis, and diagnostic ML. Built five
independent technical case studies: two on real clinical data (breast cancer
diagnosis and hospital re-identification risk) and three simulated
evaluation designs for AI safety, LLM review, and retrieval-augmented
generation. Strong
at framing questions, designing evaluations, quantifying uncertainty, and
communicating results and limitations to technical and non-technical audiences.

## Technical skills

**Data and statistics:** Python, R, SQL, Pandas, Polars, NumPy, SciPy, statsmodels; experimental design, power analysis, hypothesis testing, bootstrap confidence intervals, inter-rater reliability, Bayesian hierarchical modeling, MCMC diagnostics, calibration

**Machine learning:** PyTorch, scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, SMOTE, SHAP; feature engineering, model selection, cross-validation, threshold analysis, explainability

**LLM and systems:** OpenAI, Anthropic, Hugging Face, LangChain, FastAPI, MLflow, Qdrant, BM25, ColBERT, Docker, Kubernetes; LLM evaluation, retrieval quality, observability design, cost/latency trade-offs

## Education

**M.S., Applied Statistics** — Rochester Institute of Technology, expected 2026
Coursework: Bayesian methods, machine learning, experimental design, deep
learning, statistical learning theory, and computational statistics.

## Selected technical projects

### Breast Cancer Classification (two-semester RIT capstone, extended)

Independent research case study · April 2026

- Benchmarked eight ensemble classifiers on the Wisconsin Diagnostic Breast Cancer dataset, whose features are cell-nucleus measurements from digitized fine needle aspirate images.
- Reached 99.12% held-out accuracy (113/114) and 0.9987 ROC-AUC; traced the single error (a benign tumor flagged as malignant) and confirmed no malignant case was missed.
- Added probability calibration, threshold analysis, and per-prediction SHAP explanations; documented why a curated single-center benchmark is not clinical validation.

**Methods:** scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, SMOTE, SHAP, MLflow, FastAPI

### Clinical Privacy vs. Predictive Utility

Independent research case study · 2026

- Measured re-identification risk on 101,766 real hospital encounters (Diabetes 130-US) with k-anonymity, l-diversity, t-closeness, and a differential-privacy utility curve.
- Showed generalization halves unique-record risk (15.4% → 7.6%) for a 0.3-point AUC cost in 30-day readmission prediction; suppression costs 31% of records and 2.2 points.
- Found the model's AUC falls to 0.515 for patients aged 90+ despite 0.672 overall, and that the default threshold gives 0.2% recall; used patient-level splits to prevent leakage.

**Methods:** scikit-learn, HistGradientBoosting, permutation importance, privacy metrics, Zerve

### AI Safety Red-Team Evaluation

Independent research case study · April 2026

- Designed a two-stage evaluation workflow that combines LLM ensemble labels with supervised harm classification across 12,500 response pairs and six harm categories.
- In a simulated evaluation, reported α = 0.81 inter-rater reliability and 96.8% held-out classification accuracy; separated agreement, model performance, and uncertainty analysis.
- Used Bayesian hierarchical modeling and feature attribution to support risk review and audit-oriented reporting.

**Methods:** LLM evaluation, XGBoost/stacking, PyMC, SHAP, MLflow

### LLM Ensemble Textbook Bias Detection

Independent research case study · April 2026

- Designed a rubric-based LLM ensemble analysis and demonstrated it on a simulated corpus of 4,500 passages and 67,500 ratings for reliability and uncertainty analysis.
- Reported Krippendorff's α = 0.84 and modeled publisher-level effects with Bayesian partial pooling and MCMC diagnostics.
- Documented an expert-review-oriented workflow for interpreting disagreement and high-uncertainty passages.

**Methods:** LLM-as-judge, PyMC, ArviZ, FastAPI, MLflow

### RAG Production Pipeline

Independent systems-design case study · April 2026

- Designed a hybrid RAG architecture using dense retrieval, BM25, re-ranking, grounding checks, and confidence calibration.
- Reported 96.3% Recall@10 and 94.2% citation precision in a simulated evaluation, with latency, throughput, drift, and failure-mode analysis.
- Specified observability, privacy, and rollback considerations needed before a production deployment.

**Methods:** Qdrant, BM25, ColBERT, OpenAI, Kafka, Kubernetes, Prometheus, MLflow

## Technical reports

| Technical report | Version | Date |
|---|---:|---|
| [AI Safety Red-Team Evaluation](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) | 2.0.0 | April 2026 |
| [LLM Ensemble Textbook Bias Detection](./LLM_Ensemble_Bias_Detection_Report.md) | 4.0.0 | April 2026 |
| [Breast Cancer Classification](./Breast_Cancer_Classification_Report.md) | 4.0.0 | April 2026 |
| [RAG Production Pipeline](./RAG_Project_Report.md) | 3.0.0 | April 2026 |

**Availability:** Long Island, NY · On-site, hybrid, or remote · Authorized to work in the United States
