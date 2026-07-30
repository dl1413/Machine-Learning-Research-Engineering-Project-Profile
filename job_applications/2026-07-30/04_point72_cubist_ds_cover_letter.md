# Cover Letter — Point72 (Cubist), Data Scientist

**Role:** Data Scientist, Cubist Systematic Strategies
**Location:** New York, NY
**Apply:** https://careers.point72.com/
**Lead project:** LLM Ensemble Textbook Bias Detection (Bayesian rigor + hypothesis testing)

---

Dear Cubist / Point72 hiring team,

The role — working with the quantitative research team to engineer, validate, and refine features that feed systematic models across equities, futures, and FX — reads like the applied side of the statistics MS I'm finishing. The three projects I'd bring in translate cleanly to systematic feature validation.

**Bayesian inference and hypothesis testing at production scale.** My most relevant project is a multi-LLM bias detection study I published in April 2026. It processes **67,500 ratings across 4,500 passages** through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble. The interesting layer for a systematic-quant team is the statistics: PyMC hierarchical model with partial pooling, **MCMC diagnostics R-hat < 1.01 and ESS > 1000**, 95% HDI credible intervals per group. Group-level signal came in at **Friedman χ² = 42.73, p < 0.001**, with Nemenyi post-hoc pairwise comparisons and Bonferroni / FDR corrections on 20+ contrasts. The 12.3% highest-uncertainty passages were flagged for review rather than pushed to the point estimate — the same discipline that keeps a feature from looking real when it isn't.

**Feature engineering and ensemble modeling.** My clinical-grade classification project ran an 8-algorithm ensemble benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) with nested cross-validation, VIF multicollinearity pruning, SMOTE balancing, and RFE selection. Best model: **99.12% accuracy, ROC-AUC 0.9987, cross-val stability 98.46% ± 1.12%**. Bayesian hyperparameter search with Optuna TPE converged in 5× fewer trials than grid search (45 vs 240) — the kind of compute-efficient tuning that scales when you're screening features under budget.

**Production data pipelines.** An LLM red-team evaluation framework at **850 samples/hour, $0.018/sample, α = 0.81** across 12,500 pairs, with circuit breakers, exponential backoff, MLflow experiment tracking, and SHAP explanations. Same infrastructure hygiene you'd want on a nightly feature-validation batch.

I'm completing an MS in Applied Statistics at RIT (Bayesian inference, experimental design, high-dimensional statistics), targeting a 2026 start, based in / available for NYC on-site, US work authorized.

Portfolio and full technical reports: github.com/dl1413.

Best,
Derek Lankeaux
