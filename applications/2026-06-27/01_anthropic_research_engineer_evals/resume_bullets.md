# Resume bullets — Anthropic, Research Engineer, Model Evaluations

Drop these into the "Research Projects" section. Lead with Project 1.

## AI Safety Red-Team Evaluation Framework (LEAD)
*Independent Research | January 2026*

- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**
- Cut human-eval cost **340x ($6.12 → $0.018/sample)** while maintaining Krippendorff's alpha = 0.81 across rater models
- Built production eval harness processing **850 samples/hr** with circuit breakers, exponential backoff, async batching, and MLflow run tracking — re-runnable end-to-end from any commit
- Stacked 3 LLM judges into an XGBoost meta-classifier over 47 engineered harm-signal features (linguistic / semantic / structural)
- Quantified judge disagreement via PyMC Bayesian hierarchical model with 95% HDI risk intervals, surfacing systematic per-model blind spots
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP per-prediction explanations and full provenance trails

## LLM Ensemble Bias Detection (Supporting)
*Independent Research | Q1 2026*

- Operated 3-LLM ensemble at production scale: **67,500 ratings over 4,500 passages (2.5M tokens)** at Krippendorff's alpha = 0.84
- Surfaced statistically significant publisher-level bias (Friedman chi-squared = 42.73, p < 0.001) in 3 of 5 publishers via Bayesian partial-pooling hierarchical model (R-hat < 1.01, ESS > 1000)
- Validated rubric stability with 92% pairwise inter-LLM correlation

## Clinical-Grade Breast Cancer Classification (Supporting)
*Independent Research | 2026*

- Benchmarked 8 algorithms under nested cross-validation; shipped winning stacking ensemble at **99.12% accuracy, 100% precision, ROC-AUC 0.9987**
- Productionized behind FastAPI with MLflow model registry; <100ms p95 latency

## JD keyword echo (for ATS)
LLM evaluation, model evaluations, eval harness, LLM-as-judge, Claude,
red-team, harm classification, RLHF data quality, Constitutional AI,
Python, MLflow, evaluation infrastructure, capability evals, jailbreak.
