# Derek Lankeaux, MS

**Machine Learning Research Engineer | LLM Evaluation Specialist | AI Safety Researcher**

Greenlawn, NY 11740 · (631) 626-0511 · dl1413@g.rit.edu
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Impact-driven Machine Learning Research Engineer specialized in frontier-LLM evaluation, multi-model ensemble systems, and Bayesian inference pipelines. Currently evaluating frontier models (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, DeepSeek-R1) in production at Toloka AI and developing LLM-powered features at Handshake. Proven track record delivering 96.8-99.12% accuracy models with rigorous statistical validation. Experienced processing 10K+ daily inferences with research-grade reliability (Krippendorff's alpha >= 0.81). Published researcher with 3 technical reports demonstrating compliance with IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act standards.

---

## Core Competencies

**LLM Systems & Evaluation:** Frontier model evaluation (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, DeepSeek-R1), Constitutional AI compliance, LLM-as-Judge, agentic workflow evaluation, prompt engineering, few-shot learning, chain-of-thought reasoning, NL2SQL, RAG pipeline evaluation

**Statistical Methods:** Bayesian hierarchical modeling (PyMC/ArviZ), MCMC sampling (NUTS, HMC), convergence diagnostics (R-hat, ESS), A/B testing, hypothesis testing, time series, survival analysis

**ML Engineering & Production:** Ensemble methods, PyTorch, TensorFlow, scikit-learn, SHAP/LIME, SMOTE, RFE, cross-validation, hyperparameter optimization (Optuna TPE), Fairlearn

**MLOps & Infrastructure:** Python, R, SQL, FastAPI, Flask, Docker, Kubernetes, MLflow, LangChain, GitHub Actions, CI/CD, vector databases, Redis

---

## Professional Experience

### AI Quality Assurance Specialist | Toloka AI (Mindrift)
*Oct 2025 – Present*

- Evaluate frontier LLMs (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4) processing 10,000+ daily inferences across reasoning, instruction-following, and factual accuracy
- Implement harm detection achieving <5% false negative rate across 6 categories
- Design autonomous AI agent behavior assessments and Constitutional AI compliance metrics; identify adversarial patterns and edge cases
- Provide annotations integrated into fine-tuning pipelines with MLflow tracking

### AI Trainer | Handshake
*Nov 2025 – Present*

- Develop LLM-powered features for career platform serving 20M+ students; implement prompt engineering and few-shot learning to evaluate AI models with LLM-as-Judge methodologies
- Conduct evaluation of frontier models (GPT-5.1, Claude Opus 4.5) for production deployment
- Design A/B testing frameworks and analyze engagement metrics for ML-driven features

### Cloud Software Analyst | BRdata Software Solutions
*Apr 2024 – Nov 2024*

- Developed NL2SQL training dataset using LangChain + Groq API through chain-of-thought reasoning and few-shot learning; processed 180 samples/hour with Redis caching
- Engineered backend pipelines integrating SQL Server with cloud architecture, achieving 10% latency reduction
- Orchestrated Docker deployment improving reliability by 20%

### Mathematics Instructor | Huntington Learning Center
*Oct 2021 – Dec 2023*

- Delivered data-driven instruction to 30+ students; increased standardized test scores up to 150 points through curriculum interventions
- Maintained 95%+ satisfaction rating

---

## Research Projects

### AI Safety Red-Team Evaluation Framework
*Oct 2025 – Dec 2025*

- **Stage 1:** Architected dual-stage harm detection system processing 12,500 adversarial pairs across 12 attack categories; three-model ensemble achieving Krippendorff's alpha = 0.81, 340x cost reduction ($0.018/sample vs $6.12), 850 samples/hour
- **Stage 2:** Engineered 5,800+ dimensional feature space; Stacking Classifier achieved 96.8% accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923
- Bayesian uncertainty quantification, continuous monitoring (KS test), SHAP explainability per IEEE 2830-2025
- Implemented Bayesian hierarchical modeling for multi-model risk analysis with 95% HDI

**Tech Stack:** GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, XGBoost, Stacking Classifier, PyMC, SHAP, MLflow, Constitutional AI

---

### LLM Ensemble Textbook Bias Quantification
*Jun 2025 – Sep 2025*

- Quantified political bias in 150 textbooks using multi-LLM pipeline processing 67,500 ratings across 4,500 passages; achieved Krippendorff's alpha = 0.84 with pairwise correlations r = 0.87–0.92
- Built Bayesian hierarchical model with partial pooling (R-hat < 1.01, ESS > 3,000); identified systematic bias in 60% of publishers
- Engineered production-grade API integration with circuit breakers, exponential backoff, and MLflow experiment tracking
- Delivered research-quality technical report with 95% HDI quantification and reproducible statistical methodology

**Tech Stack:** GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, PyMC, ArviZ, MLflow, FastAPI, LangChain

---

### NL2SQL Production System with LangChain
*Apr 2024 – Nov 2024*

- Deployed conversational SQL interface for retail analytics achieving 90% query accuracy through few-shot learning, chain-of-thought reasoning, and schema-aware generation
- Built on FastAPI + Docker + Redis with security protocols

**Tech Stack:** LangChain, Groq API, FastAPI, Docker, Redis, SQL Server

---

### Ensemble Learning for Breast Cancer Classification
*May 2023 – Dec 2023*

- Benchmarked 8 ensemble algorithms; AdaBoost achieved 99.12% accuracy, 100% precision, ROC-AUC 0.9987
- Deployed FastAPI endpoint with MLflow, SHAP explainability, and Fairlearn auditing per IEEE 2830-2025
- Achieved <100ms p95 latency in production
- Applied advanced preprocessing pipeline with VIF multicollinearity analysis, SMOTE class balancing, and RFE feature selection

**Tech Stack:** scikit-learn, XGBoost, LightGBM, AdaBoost, SMOTE, SHAP, Fairlearn, MLflow, FastAPI

---

## Education

**Master of Science in Applied Statistics**
Rochester Institute of Technology *(in progress)*
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

**Bachelor of Science in Applied Mathematics & Statistics**
Stony Brook University

**Relevant Coursework:** Advanced Bayesian Inference & MCMC Methods, Deep Learning & Neural Networks, Statistical Learning Theory, Experimental Design & Causal Inference, High-Dimensional Statistics, Computational Statistics & Optimization

---

## Key Achievements

- Built LLM Red-Team Framework with 3-model ensemble achieving 340x cost reduction ($0.018/sample) and audit-grade reliability (alpha = 0.81)
- Developed Multi-Model Evaluation Pipeline with Krippendorff's alpha = 0.81-0.84 across frontier LLMs with Bayesian uncertainty quantification
- Deployed Clinical-Grade ML System achieving 99.12% accuracy exceeding human expert performance (90-95%)
- Currently scaling production NLP workflows at Toloka AI (10K+ daily inferences) and Handshake (20M+ student platform)
- Engineered low-latency inference with FastAPI deployments achieving <100ms p95 latency
- Published 3 research-quality technical reports with p < 0.001 significance, 95% HDI intervals, and SHAP explainability

---

## Publications & Technical Reports

| Title | Type | Date |
|-------|------|------|
| AI Safety Red-Team Evaluation | Technical Report | Dec 2025 |
| LLM Ensemble Textbook Bias Quantification | Technical Report | Sep 2025 |
| Breast Cancer Classification | Technical Report | Dec 2023 |

---

**Location:** Greenlawn, NY (Long Island) · Open to NYC, hybrid, or remote
**Work Authorization:** Authorized to work in the United States
