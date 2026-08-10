# Canonical Facts — Use These In Every Application

> Source of truth for cover letters, resume bullets, and outreach. The templates in this folder were drafted from the in-repo README, which is stale (uses older model names and pre-evolution project numbers). When you use a template, **substitute these current facts** before sending.

Last updated: From `Derek_Lankeaux_Resume_2026_6.docx` (May 2026).

---

## Contact

- **Name:** Derek Lankeaux
- **Location:** Greenlawn, NY 11740 (Long Island — NYC commutable)
- **Phone:** (631) 626-0511
- **Email:** dl1413@g.rit.edu
- **LinkedIn:** linkedin.com/in/derek-lankeaux
- **GitHub:** github.com/dl1413
- **Work auth:** US authorized

---

## Education

- **MS, Applied Statistics** — Rochester Institute of Technology (in progress)
- **BS, Applied Mathematics & Statistics** — Stony Brook University

---

## Current & Past Roles

| Role | Org | Dates |
|---|---|---|
| AI Quality Assurance Specialist | Toloka AI (Mindrift) | Oct 2025 – Present |
| AI Trainer | Handshake | Nov 2025 – Present |
| Cloud Software Analyst | BRdata Software Solutions | Apr 2024 – Nov 2024 |
| Mathematics Instructor | Huntington Learning Center | Oct 2021 – Dec 2023 |

**Toloka AI signal:** Evaluate frontier LLMs processing 10,000+ daily inferences across reasoning, instruction-following, factual accuracy; harm detection with <5% false negative rate across 6 categories; Constitutional AI compliance; MLflow-tracked annotations integrated into fine-tuning.

**Handshake signal:** Develop LLM-powered features for a career platform serving 20M+ students; prompt engineering, few-shot learning, LLM-as-Judge evaluation; A/B testing frameworks and engagement metrics analysis.

**BRdata signal:** NL2SQL training dataset using LangChain + Groq API; chain-of-thought reasoning and few-shot learning; 180 samples/hour with Redis caching. Backend pipelines integrating SQL Server with cloud architecture — 10% latency reduction, 20% reliability improvement via Docker orchestration.

**Huntington signal:** Data-driven instruction to 30+ students; raised test scores up to 150 points via curriculum interventions; 95%+ satisfaction.

---

## Tech Stack (current naming)

- **Frontier models:** GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, DeepSeek-R1
- **LLM patterns:** LLM-as-Judge, Constitutional AI compliance, agentic workflow eval, prompt engineering, few-shot, chain-of-thought, NL2SQL, RAG pipeline eval
- **Statistics:** Bayesian hierarchical modeling (PyMC/ArviZ), MCMC (NUTS, HMC), convergence diagnostics (R-hat, ESS), A/B testing, hypothesis testing, time series, survival analysis
- **ML eng:** Ensemble methods, PyTorch, TensorFlow, scikit-learn, SHAP/LIME, SMOTE, RFE, cross-validation, hyperparameter optimization (Optuna TPE)
- **MLOps:** Python, R, SQL, FastAPI, Flask, Docker, Kubernetes, MLflow, LangChain, GitHub Actions, CI/CD, vector databases

⚠️ The older templates in this folder still reference **GPT-4o, Claude-3.5, Llama-3.2** — swap to current names above before sending.

---

## Project 1 — AI Safety Red-Team Evaluation Framework

**Dates:** Oct 2025 – Dec 2025

**Stage 1 (LLM ensemble annotation):**
- Dual-stage harm detection system
- 12,500 adversarial pairs
- **12 attack categories**
- Three-model ensemble
- **Krippendorff's α = 0.81**
- **340× cost reduction** ($0.018/sample vs $6.12 human baseline)
- **850 samples/hour** throughput

**Stage 2 (ML classifier):**
- **5,800+ dimensional feature space** (⚠️ NOT 47 features — that's stale from the in-repo README)
- Stacking Classifier
- **96.8% accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923**
- Bayesian uncertainty quantification
- Continuous monitoring (KS test for drift)
- SHAP explainability
- IEEE 2830-2025 compliance

---

## Project 2 — LLM Ensemble Textbook Bias Quantification

**Dates:** Jun 2025 – Sep 2025

- **150 textbooks** analyzed (⚠️ this scale was missing from older templates)
- Multi-LLM pipeline
- **67,500 ratings** across **4,500 passages**
- **Krippendorff's α = 0.84**
- Pairwise correlations **r = 0.87–0.92**
- Bayesian hierarchical model with partial pooling
- **R-hat < 1.01, ESS > 3,000**
- **Systematic bias identified in 60% of publishers**

---

## Project 3 — Ensemble Learning for Breast Cancer Classification

**Dates:** May 2023 – Dec 2023 *(note: older than the LLM projects)*

- Benchmarked **8 ensemble algorithms**
- AdaBoost won: **99.12% accuracy, 100% precision, ROC-AUC 0.9987**
- FastAPI endpoint deployment
- MLflow registry
- SHAP explainability + **Fairlearn** auditing per IEEE 2830-2025
- **<100ms p95 latency**

---

## Bonus Project — NL2SQL Production System with LangChain

**Dates:** Apr 2024 – Nov 2024 (concurrent with BRdata role)

- Conversational SQL interface for retail analytics
- **90% query accuracy** via few-shot, chain-of-thought, schema-aware generation
- FastAPI + Docker + Redis stack with security protocols

*Use this as a 4th project bullet for roles emphasizing LangChain, RAG, NL2SQL, or production LLM deployment.*

---

## Recurring substitutions when editing the templates in this folder

| What the template says (stale) | What to write (current) |
|---|---|
| "January 2026" / "Independent Research Project" | Use the actual project date range and either "Independent Research" or just the dates |
| GPT-4o | GPT-5.1 |
| Claude-3.5 / Claude-3.5-Sonnet | Claude Opus 4.5 |
| Llama-3.2 | Llama 4 |
| "6 harm categories" (Red-Team) | "12 attack categories" |
| "47 engineered features" (Red-Team) | "5,800+ dimensional feature space" |
| "3/5 publishers" (Bias) | "60% of publishers analyzed" or "60% of 150 textbooks across publishers" |
| "Available for remote/hybrid" | "Based in Greenlawn, NY (Long Island) · open to NYC/hybrid/remote" |
| Project bullets framed as "independent" | When useful, mention concurrent work at Toloka AI / Handshake for employment continuity |
