# Derek Lankeaux

**Data Scientist | Applied Statistician | LLM Evaluation & Applied ML**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Summary

Applied Statistics M.S. candidate focused on experimental design, Bayesian
inference, and practical machine-learning evaluation. Built four independent
technical case studies spanning AI-safety evaluation, uncertainty-aware LLM
review, diagnostic-ML benchmarking, and retrieval-augmented generation. Strong
at framing questions, designing evaluations, quantifying uncertainty, and
communicating results and limitations to technical and non-technical audiences.

## Technical skills

**Data and statistics:** Python, R, SQL, Pandas, Polars, NumPy, SciPy, statsmodels; experimental design, power analysis, hypothesis testing, bootstrap confidence intervals, inter-rater reliability, Bayesian hierarchical modeling, MCMC diagnostics, calibration

**Machine learning:** scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, SMOTE, SHAP; feature engineering, model selection, cross-validation, threshold analysis, explainability

**LLM and systems:** OpenAI, Anthropic, Hugging Face, LangChain, FastAPI, MLflow, Qdrant, BM25, ColBERT, Docker, Kubernetes; LLM evaluation, retrieval quality, observability design, cost/latency trade-offs

## Education

**M.S., Applied Statistics** — Rochester Institute of Technology, expected 2026
Coursework: Bayesian methods, machine learning, experimental design, deep
learning, statistical learning theory, and computational statistics.

## Selected technical projects

### AI Safety Red-Team Evaluation

Independent research case study · April 2026

- Designed a two-stage evaluation workflow that combines LLM ensemble labels with supervised harm classification across 12,500 response pairs and six harm categories.
- Reported α = 0.81 inter-rater reliability and 96.8% held-out classification accuracy; separated agreement, model performance, and uncertainty analysis in the evaluation.
- Used Bayesian hierarchical modeling and feature attribution to support risk review and audit-oriented reporting.

**Methods:** LLM evaluation, XGBoost/stacking, PyMC, SHAP, MLflow

### LLM Ensemble Textbook Bias Detection

Independent research case study · April 2026

- Evaluated 4,500 textbook passages with a rubric-based LLM ensemble, generating 67,500 ratings for reliability and uncertainty analysis.
- Reported Krippendorff's α = 0.84 and modeled publisher-level effects with Bayesian partial pooling and MCMC diagnostics.
- Documented an expert-review-oriented workflow for interpreting disagreement and high-uncertainty passages.

**Methods:** LLM-as-judge, PyMC, ArviZ, FastAPI, MLflow

### Breast Cancer Classification Benchmark

Independent research case study · April 2026

- Benchmarked eight ensemble classifiers on the Wisconsin Diagnostic Breast Cancer dataset with preprocessing, feature selection, calibration, and cross-validation.
- Reported 99.12% held-out accuracy and 0.9987 ROC-AUC for the best configuration, together with calibration and threshold analyses.
- Framed the work as a diagnostic decision-support benchmark; it is not a clinical device or patient-validation study.

**Methods:** scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, SMOTE, SHAP

### RAG Production Pipeline

Independent systems-design case study · April 2026

- Designed a hybrid RAG architecture using dense retrieval, BM25, re-ranking, grounding checks, and confidence calibration.
- Reported 96.3% Recall@10 and 94.2% citation precision in the documented evaluation, with latency, throughput, drift, and failure-mode analysis.
- Specified observability, privacy, and rollback considerations needed before a production deployment.

**Methods:** Qdrant, BM25, ColBERT, OpenAI, Kafka, Kubernetes, Prometheus, MLflow

## Publications

| Technical report | Version | Date |
|---|---:|---|
| [AI Safety Red-Team Evaluation](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) | 2.0.0 | April 2026 |
| [LLM Ensemble Textbook Bias Detection](./LLM_Ensemble_Bias_Detection_Report.md) | 4.0.0 | April 2026 |
| [Breast Cancer Classification](./Breast_Cancer_Classification_Report.md) | 4.0.0 | April 2026 |
| [RAG Production Pipeline](./RAG_Project_Report.md) | 3.0.0 | April 2026 |

**Availability:** Remote/hybrid · Seeking 2026 opportunities · Authorized to work in the United States
