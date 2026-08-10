# Datadog — Senior Applied Scientist, Large Language Models / Generative AI

**Posting:** https://careers.datadoghq.com/detail/5065446/
**Location:** New York, NY (hybrid)
**Date drafted:** 2026-07-08

---

Dear Datadog Applied AI team,

I'm applying for the Senior Applied Scientist role on your Large Language Models / Generative AI team. The team's remit — fine-tuning, evaluating, and shipping LLM features into a monitoring product where latency, cost, and reliability all count — lines up directly with the LLM-ensemble evaluation work I've been publishing this year.

**Where I can contribute on day one:**

- **Offline LLM evaluation at production scale.** My LLM Ensemble Bias Detection system processed **67,500 ratings across 4,500 passages, 2.5M tokens** through GPT-4o, Claude-3.5, and Llama-3.2 with 92% pairwise correlation, then modeled the results with a PyMC Bayesian hierarchical model (partial pooling, MCMC R-hat < 1.01, 95% HDI) to surface *credible* between-source differences rather than raw averages — the kind of judge-ensemble + statistical rigor that keeps a benchmark trustworthy as prompts and models drift.
- **LLM-as-judge + human-in-the-loop, calibrated for cost.** My AI Safety Red-Team pipeline paired an LLM ensemble (α = 0.81) with a downstream Stacking Classifier over 47 engineered features to reach **96.8% accuracy at $0.018/sample vs $6.12 for human annotation (340× cost reduction) and 850 samples/hour** — the same pattern that lets a product team run per-release regression evals without blowing the budget.
- **Production hygiene, not notebook demos.** MLflow tracking, circuit breakers, exponential backoff, SHAP explanations, model cards aligned with IEEE 2830-2025 / ISO/IEC 23894:2025. FastAPI serving at <100ms p95 on a calibrated clinical classifier (Platt scaling, ECE 0.0089) — I know how to hand a model to platform engineering without loose ends.

**Fit to Datadog specifically:** LLM features in a monitoring product live or die on eval loops that catch regressions before customers do. I'd want to spend the first 60 days on the eval harness — coverage of your existing prompt/response fixtures, calibration diagnostics per model version, and a Bayesian pass/fail rule that keeps false-alarm rates honest as you cycle base models.

Portfolio and code: https://dl1413.github.io/LLM-Portfolio/ • https://github.com/dl1413. Resume attached.

Thank you,
Derek Lankeaux
