# Derek Lankeaux, MS

**Machine Learning Research Engineer | LLM Evaluation Specialist | AI Safety Researcher**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Impact-driven Machine Learning Research Engineer specialized in building production-grade LLM evaluation frameworks, multi-model ensemble systems, and Bayesian inference pipelines. Proven track record delivering 96.8-99.12% accuracy models with rigorous statistical validation. Experienced processing 80K+ LLM annotations at production scale (850 samples/hr) while maintaining research-grade reliability (Krippendorff's alpha >= 0.81). Published researcher with 3 technical reports demonstrating compliance with IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act standards.

---

## Technical Skills

**Languages:** Python 3.12+, R, SQL, Bash

**ML Frameworks:** PyTorch 2.0+, TensorFlow 2.15+, scikit-learn 1.5+, JAX

**LLM APIs:** OpenAI (GPT-4o), Anthropic (Claude-3.5), Meta (Llama-3.2), HuggingFace

**Ensemble ML:** XGBoost 2.1+, LightGBM 4.5+, CatBoost, AdaBoost

**Bayesian Statistics:** PyMC 5.15+, ArviZ 0.18+, NumPyro, Stan

**Data Stack:** Pandas 2.2+, Polars 1.0+, NumPy 2.0+, Dask, Apache Arrow

**MLOps:** MLflow 2.15+, Weights & Biases, DVC, Kubeflow

**Deployment:** FastAPI 0.110+, Docker, Kubernetes, AWS, GCP

**Explainability:** SHAP, LIME, Captum, InterpretML

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

**LLM & AI Safety:** Multi-model ensemble architectures, AI safety red-team evaluation, Prompt engineering & optimization, Inter-rater reliability analysis, Harm detection & classification, API integration at scale

**Statistical ML:** Ensemble methods (8+ algorithms), Bayesian hierarchical modeling, MCMC diagnostics (R-hat, ESS), Hypothesis testing & validation, Feature engineering & selection

**Production MLOps:** FastAPI model deployment, MLflow experiment tracking, Circuit breakers & rate limiting, Monitoring & drift detection, Docker/Kubernetes orchestration

---

**Location:** Available for remote/hybrid positions  
**Timeline:** Seeking positions starting 2026  
**Work Authorization:** Authorized to work in the United States
