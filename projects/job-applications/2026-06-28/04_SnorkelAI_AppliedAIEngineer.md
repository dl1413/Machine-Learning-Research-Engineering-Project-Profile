# Snorkel AI — Applied AI Engineer, AI Solutions

**Posting:** https://job-boards.greenhouse.io/snorkelai/jobs/5709067004
**Location:** Remote / Hybrid
**Why fit:** Snorkel's wheelhouse — weak supervision, programmatic labeling, evaluation harnesses — is exactly the gap my LLM-ensemble + ML-classifier two-stage pipeline closes.

---

Dear Snorkel AI team,

I'm applying for the Applied AI Engineer role. Snorkel's bet — that programmatic labeling and rigorous evaluation beat brute-force human annotation — is the same bet I made in my portfolio, and the receipts are quantitative.

My **AI Safety Red-Team Evaluation** project is essentially a Snorkel-style LF stack with LLMs as the labeling functions. A GPT-4o / Claude-3.5 / Llama-3.2 ensemble annotates 12,500 response pairs across 6 harm categories with Krippendorff's α = 0.81; a downstream Stacking Classifier on 47 engineered features turns those weak labels into 96.8% accuracy / 97.2% precision / 0.9923 ROC-AUC at 850 samples/hour. Cost dropped from $6.12 / sample to $0.018 — a 340× reduction that survives SHAP-based audit. This is the production pattern Snorkel customers actually want.

**LLM Ensemble Textbook Bias Detection** adds the statistical layer: PyMC hierarchical model with partial pooling, R-hat < 1.01, Friedman χ² = 42.73 (p < 0.001), publisher-level credible bias with 95% HDI, and bootstrap CIs flagging 12.3% of passages for human review. That's the LF agreement / coverage / reliability triage Snorkel Flow surfaces, with the math to back it.

**Breast Cancer Classification** is the engineering bookend — Optuna TPE (converged in 45 trials vs 240 for grid), Platt calibration (ECE 0.0089), FastAPI under 100ms p95, MLflow registry.

I ship clean pipelines with circuit breakers, exponential backoff, MLflow tracking, and model cards aligned to IEEE 2830-2025 — the discipline customer-facing deployments need.

Would love to talk. Portfolio: https://dl1413.github.io/LLM-Portfolio/

Best,
Derek Lankeaux
