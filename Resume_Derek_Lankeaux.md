# Derek Lankeaux, MS

**Data Scientist | Applied Statistician | LLM Evaluation & GenAI Specialist**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Data Scientist with an Applied Statistics MS focused on experimentation, Bayesian inference, and applied machine learning. Two portfolio projects: (1) an LLM-as-judge evaluation pipeline with Bayesian hierarchical inference (67,500 ratings, Krippendorff's alpha = 0.84, R-hat < 1.01, p < 0.001) and (2) a calibrated binary classifier with decision-policy tuning (99.12% accuracy, ECE = 0.0089, two operating points tied to asymmetric cost ratios). Comfortable owning the full data science workflow: framing the question, designing the experiment, writing the SQL, building the model, quantifying uncertainty, and communicating impact to non-technical stakeholders. Both projects shipped with model cards, posterior plots, and SHAP attributions, aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.

---

## Technical Skills

**Languages:** Python 3.12+, R, SQL, Bash

**Experimentation & Statistics:** A/B testing, power analysis, hypothesis testing, multiple-testing correction (Bonferroni, FDR), effect sizes (Cohen's d, eta-squared), bootstrap CIs, causal & quasi-experimental design, inter-rater reliability (Krippendorff's alpha, Cohen's kappa)

**Bayesian Statistics:** PyMC 5.15+, ArviZ 0.18+, NumPyro, Stan; hierarchical models, MCMC diagnostics (R-hat, ESS), 95% HDI, model calibration

**Modeling & ML:** scikit-learn 1.5+, XGBoost 2.1+, LightGBM 4.5+, CatBoost, AdaBoost, PyTorch 2.0+, TensorFlow 2.15+; feature engineering, calibration (Platt, isotonic), threshold tuning

**GenAI / LLM:** OpenAI (GPT-4o), Anthropic (Claude-3.5), Meta (Llama-3.2), HuggingFace, LangChain; LLM-as-judge, multi-model ensembles, prompt iteration, offline evaluation

**Data Stack:** SQL, Pandas 2.2+, Polars 1.0+, NumPy 2.0+, Dask, Apache Arrow

**MLOps & Deployment:** MLflow 2.15+, Weights & Biases, DVC, FastAPI 0.110+, Docker, Kubernetes, AWS, GCP

**Explainability & Responsible AI:** SHAP, LIME, Captum, InterpretML; model cards, IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act

---

## Education

**Master of Science in Applied Statistics**  
Rochester Institute of Technology | Expected 2026  
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

**Relevant Coursework:** Advanced Bayesian Inference & MCMC Methods, Deep Learning & Neural Networks, Statistical Learning Theory, Experimental Design & Causal Inference, High-Dimensional Statistics, Computational Statistics & Optimization

---

## Data Science Projects

### LLM Ensemble Textbook Bias Detection — GenAI Evaluation + Bayesian Inference
*Independent Research Project | 2026*

- Designed an LLM-as-judge evaluation pipeline using three frontier models (GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B) to rate 4,500 textbook passages, producing 67,500 ratings across 2.5M tokens
- Validated multi-rater agreement with Krippendorff's alpha = 0.84 (excellent, >=0.80) and pairwise Pearson correlations of 0.87-0.92 before running any downstream inference
- Fit a Bayesian hierarchical model in PyMC with partial pooling on publishers and passages; MCMC converged cleanly (R-hat < 1.01, ESS > 3,000) and produced 95% HDIs for each publisher
- Confirmed effects with a non-parametric Friedman test (chi-squared = 42.73, p < 0.001); three of five publishers showed credible bias (95% HDI excluding zero), with effects from -0.48 to +0.38 on a [-2, +2] scale
- Built an uncertainty-aware triage workflow: bootstrap CIs at the passage level flagged 12.3% of passages as high-uncertainty for human review, turning the pipeline into an LLM-plus-human hybrid
- Engineered the production layer with circuit breakers, exponential backoff, deterministic cached re-runs, MLflow experiment tracking, and a stakeholder-facing per-publisher report card

**Tech Stack:** Python, GPT-4o, Claude-3.5-Sonnet, Llama-3.2, PyMC, ArviZ, SciPy, statsmodels, Pandas, MLflow, FastAPI, LangChain

---

### Calibrated Binary Classifier with Decision-Policy Tuning (WDBC)
*Independent Research Project | 2026*

- Framed the problem as a high-stakes binary classification task with asymmetric error costs (missed cancer vs. unnecessary biopsy) rather than a single accuracy target
- Benchmarked eight ensemble algorithms (Random Forest, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM, Voting, Stacking) under identical stratified k-fold CV; AdaBoost won at 99.12% accuracy with 10-fold CV = 98.46% +/- 1.12%
- Tuned hyperparameters with Optuna's TPE sampler, converging in 45 trials vs. ~240 for grid search (5x fewer fits for the same operating point)
- Applied Platt scaling to convert raw scores into calibrated probabilities, reducing Expected Calibration Error from 0.0312 to 0.0089 (71.5% reduction) so downstream decisions can trust the probability outputs
- Designed two operating points tied to cost ratios: threshold = 0.31 yields 100% sensitivity for mass screening; threshold = 0.62 yields 100% precision for confirmation, with the rationale documented for audit
- Built preprocessing pipeline with VIF multicollinearity analysis, SMOTE applied on training folds only, and RFE feature selection; shipped SHAP attributions and a model card aligned with IEEE 2830-2025 / EU AI Act
- Deployed via FastAPI (<100ms p95) with MLflow registry and a drift monitor on calibration ECE in addition to accuracy

**Tech Stack:** Python, scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, imbalanced-learn (SMOTE), SHAP, MLflow, FastAPI, Docker

---

## Key Achievements

- Built a GenAI evaluation pipeline that combines LLM-as-judge ensembling with Bayesian hierarchical inference, producing publisher-level findings with 95% HDIs rather than point estimates
- Designed a calibrated binary classifier with two decision policies (screening / confirmation) and a calibration-aware drift monitor — accuracy alone is insufficient in high-stakes settings
- Validated rigor end-to-end: Krippendorff's alpha = 0.84, R-hat < 1.01, p < 0.001, ECE = 0.0089 after Platt scaling, Cohen's kappa = 0.9823
- Engineered production data pipelines processing 67,500 LLM ratings / 2.5M tokens with circuit breakers, exponential backoff, deterministic re-runs, and MLflow tracking
- Communicated findings with model cards, posterior plots, calibration reliability diagrams, and per-prediction SHAP attributions — written for non-technical stakeholders
- Published 2 technical reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act

---

## Publications & Technical Reports

| Title | Type | Date |
|-------|------|------|
| LLM Ensemble Textbook Bias Detection | Technical Report v4.0.0 | 2026 |
| Calibrated Binary Classification (WDBC) | Technical Report v4.0.0 | 2026 |

---

## Core Competencies

**Experimentation & Causal Inference:** A/B test design & sample sizing, hypothesis testing, multiple-testing correction, effect-size reporting, quasi-experimental analysis, inter-rater reliability (Krippendorff's alpha, Cohen's kappa)

**Statistical Modeling:** Bayesian hierarchical modeling, MCMC diagnostics (R-hat, ESS), credible intervals (95% HDI), calibration (Platt / isotonic), ensemble methods (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting)

**GenAI / LLM Evaluation:** Multi-model ensemble architectures, LLM-as-judge & human-in-the-loop labeling, prompt engineering, offline benchmarking, cost / latency tradeoff analysis

**Production Data Science:** SQL, FastAPI model deployment, MLflow experiment tracking, circuit breakers & rate limiting, monitoring & drift detection, Docker/Kubernetes orchestration, stakeholder-ready model cards and readouts

---

**Location:** Available for remote/hybrid positions  
**Timeline:** Seeking positions starting 2026  
**Work Authorization:** Authorized to work in the United States
