# Derek Lankeaux, MS

**Entry-Level Data Engineer | Python & SQL | ETL/ELT Pipelines | Cloud-Curious**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Early-career data engineer with an Applied Statistics MS and hands-on experience building ETL/ELT pipelines in Python and SQL. Have shipped three end-to-end data projects covering multi-source API ingestion, relational data modeling, data quality checks, and BI-ready curated layers -- processing 80K+ API calls and millions of records with circuit breakers, retries, and full lineage tracking. Comfortable owning the pipeline lifecycle: extraction, transformation, validation, loading, monitoring, and documentation. Eager to grow under senior mentors on a modern cloud data stack (AWS/Azure/GCP, Airflow, dbt, Spark).

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

### Multi-Source AI Evaluation Data Pipeline
*Independent Research Project | January 2026*

- Built an end-to-end ELT pipeline in Python ingesting 12,500 records from three external LLM APIs (GPT-4o, Claude-3.5, Llama-3.2), landing raw payloads to a staging zone and modeling curated facts/dimensions for downstream analytics
- Authored SQL transformations to clean, deduplicate, and join multi-source rating data into a single analytics-ready table covering 6 harm categories and 47 derived feature columns
- Hardened ingestion with retry/backoff, circuit breakers, and idempotent writes, sustaining ~850 records/hour throughput and cutting per-record processing cost by 340x ($0.018 vs $6.12 manual)
- Implemented data quality checks (schema validation, null/range constraints, inter-rater agreement >= 0.81) to gate records before promotion from raw -> staging -> curated layers
- Versioned pipeline code in Git with reproducible runs, parameterized configs, and MLflow run tracking for lineage and auditability
- Produced documentation, run logs, and IEEE 2830-2025 / EU AI Act-aligned audit trails for cross-functional stakeholders

**Tech Stack:** Python, SQL, Pandas, REST APIs, MLflow, Git, Docker, PostgreSQL

---

### High-Volume Text Ingestion & Rating ETL
*Independent Research Project | January 2026*

- Designed and operated an ETL workflow that ingested 4,500 source documents (2.5M tokens), fanned out 67,500 rating calls across 3 external APIs, and persisted normalized results into a relational analytics schema
- Wrote Python extract/load tasks and SQL transformations (window functions, CTEs, joins) to aggregate ratings by publisher, source, and category for BI consumption
- Built reliability layer with exponential backoff, rate limiting, and circuit breakers, sustaining 80K+ outbound API calls with <0.5% failure rate
- Enforced data quality through agreement checks (Krippendorff's alpha = 0.84, 92% pairwise correlation) and automated validation before loads
- Modeled the curated layer for downstream BI dashboards (publisher-level bias metrics with 95% credible intervals) and documented schemas, run logs, and DAG dependencies
- Tracked pipeline runs and artifacts in MLflow with Git-versioned configuration for full reproducibility

**Tech Stack:** Python, SQL, Pandas, PostgreSQL, REST APIs, MLflow, FastAPI, Git

---

### Clinical Data Preparation & Feature Pipeline
*Independent Research Project | January 2026*

- Built a reproducible Python/SQL data pipeline transforming raw clinical tabular data into a feature-ready warehouse table for downstream modeling and reporting
- Implemented preprocessing stages -- type coercion, null handling, VIF-based multicollinearity pruning, SMOTE rebalancing, and RFE feature selection -- with logged data quality metrics at each step
- Authored SQL queries and Pandas transformations to produce analyst-friendly views and exported curated datasets for BI tooling
- Packaged the workflow with Docker, exposed it through a FastAPI service (<100ms p95 latency), and versioned datasets/models in MLflow registry for lineage
- Documented schemas, transformation logic, and run procedures so analysts and engineers could rerun and extend the pipeline independently
- Validated pipeline outputs against governance standards (IEEE 2830-2025) with explainability artifacts (SHAP) for downstream auditing

**Tech Stack:** Python, SQL, Pandas, scikit-learn, FastAPI, Docker, MLflow, Git

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
