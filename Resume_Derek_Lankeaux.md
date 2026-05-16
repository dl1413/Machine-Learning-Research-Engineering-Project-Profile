# Derek Lankeaux, MS

**Data Scientist | Applied Statistician | LLM Evaluation & GenAI Specialist**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Data Scientist with an Applied Statistics MS focused on experimentation, Bayesian inference, and applied machine learning. Ship end-to-end projects spanning GenAI evaluation, predictive modeling, and risk analytics — delivering 96.8-99.12% model performance with rigorous statistical validation (Krippendorff's alpha >= 0.81, MCMC R-hat < 1.01, p < 0.001). Comfortable owning the full data science workflow: framing the question, designing the experiment, writing the SQL, building the model, quantifying uncertainty, and communicating impact to non-technical stakeholders. Published 3 technical reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.

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

## Research Projects

### AI Safety Red-Team Evaluation Framework
*Independent Research Project | January 2026*

- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) achieving 96.8% accuracy in automated harm detection across 12,500 AI response pairs and 6 harm categories
- Achieved 340x cost reduction ($0.018/sample vs $6.12 human annotation) while maintaining excellent inter-rater reliability (Krippendorff's alpha = 0.81)
- Developed production ML pipeline processing 850 samples/hour with Stacking Classifier (97.2% precision, 96.1% recall, ROC-AUC 0.9923)
- Designed 47 engineered features capturing linguistic, semantic, and structural harm signals for robust classification
- Implemented Bayesian hierarchical modeling for multi-model risk analysis with 95% HDI uncertainty quantification
- Built comprehensive MLOps infrastructure with SHAP explainability, audit trails, and IEEE 2830-2025 compliance

**Tech Stack:** GPT-4o, Claude-3.5, Llama-3.2, XGBoost, Stacking Classifier, PyMC, SHAP, MLflow, Constitutional AI

---

### LLM Ensemble Textbook Bias Detection System
*Independent Research Project | January 2026*

- Built multi-LLM evaluation framework processing 67,500 bias ratings across 4,500 textbook passages with 2.5M tokens at production scale
- Achieved excellent inter-rater reliability (Krippendorff's alpha = 0.84) with 92% pairwise correlation across frontier LLMs
- Implemented Bayesian hierarchical model with partial pooling and MCMC convergence (R-hat < 1.01) for publisher-level credible bias detection
- Discovered statistically significant bias findings (Friedman chi-squared = 42.73, p < 0.001) in 3/5 publishers analyzed
- Engineered production-grade API integration with circuit breakers, exponential backoff, and MLflow experiment tracking
- Delivered research-quality technical report with 95% HDI quantification and reproducible statistical methodology

**Tech Stack:** GPT-4o, Claude-3.5, Llama-3.2, PyMC, ArviZ, MLflow, FastAPI, LangChain

---

### Clinical-Grade Breast Cancer ML Classification System
*Independent Research Project | January 2026*

- Developed ensemble ML system achieving 99.12% accuracy, exceeding human expert performance (90-95%) on breast cancer classification
- Attained 100% precision (zero false positives) and 98.59% recall with near-perfect discrimination (ROC-AUC 0.9987)
- Conducted comprehensive 8-algorithm benchmark evaluation (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting)
- Applied advanced preprocessing pipeline with VIF multicollinearity analysis, SMOTE class balancing, and RFE feature selection
- Implemented explainable AI with SHAP values for clinical transparency and fairness auditing per IEEE 2830-2025 standards
- Deployed production-ready model with MLflow registry and FastAPI serving (<100ms p95 latency)

**Tech Stack:** scikit-learn, XGBoost, LightGBM, AdaBoost, SMOTE, SHAP, MLflow, FastAPI

---

## Key Achievements

- Built LLM Red-Team Framework with 3-model ensemble achieving 340x cost reduction ($0.018/sample) and audit-grade reliability (alpha = 0.81)
- Developed Multi-Model Evaluation Pipeline with Krippendorff's alpha = 0.81-0.84 across frontier LLMs with Bayesian uncertainty quantification
- Deployed Clinical-Grade ML System achieving 99.12% accuracy exceeding human expert performance (90-95%)
- Scaled Production NLP Pipelines processing 80K+ API calls with circuit breakers, rate limiting, and MLflow experiment tracking
- Engineered Low-Latency Inference with FastAPI deployments achieving <100ms p95 latency with real-time monitoring
- Published 3 Research-Quality Technical Reports with p < 0.001 significance, 95% HDI intervals, and SHAP explainability

---

## Publications & Technical Reports

| Title | Type | Date |
|-------|------|------|
| AI Safety Red-Team Evaluation | Technical Report v1.0.0 | January 2026 |
| LLM Ensemble Textbook Bias Detection | Technical Report v3.0.0 | January 2026 |
| Breast Cancer Classification | Technical Report v3.0.0 | January 2026 |

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
