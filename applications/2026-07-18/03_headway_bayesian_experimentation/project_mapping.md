# Project → JD Mapping — Headway Bayesian Experimentation

## LLM Ensemble Textbook Bias Detection (PRIMARY MATCH)

| Their ask | My evidence |
|-----------|-------------|
| Bayesian hierarchical modeling as a default | PyMC 5.15+, partial pooling publisher → passage, R-hat < 1.01, full ESS diagnostics |
| Communicate uncertainty | 95% HDI at publisher and topic layers; 12.3% high-uncertainty passages flagged for expert review |
| Corroborate with non-parametric | Friedman χ² = 42.73, p < 0.001; Spearman correlation matrix (ρ up to 0.74) |
| Cross-team storytelling | Publication-grade technical report + model card + calibration plots |
| Scale + production hygiene | 2.5M tokens, 67,500 ratings, circuit breakers, exponential backoff, MLflow tracking |

## AI Safety Red-Team Framework (SECONDARY — multi-source uncertainty)

| Their ask | My evidence |
|-----------|-------------|
| Judge when Bayesian methods add value | Multi-model risk analysis where frequentist point estimates hide joint uncertainty; Bayesian hierarchical model with 95% HDI clarifies model-level risk |
| Reliability standards | Krippendorff's α = 0.81 across the LLM ensemble — the floor before any downstream inference |
| Turn evals into decisions | 8-vector attack taxonomy scored with defense-effectiveness delta (21.8% → 4.8%) |

## Breast Cancer Classification (TERTIARY — calibration + decision policy)

| Their ask | My evidence |
|-----------|-------------|
| Calibration & threshold policies | Platt scaling reduces ECE by 71.5% (0.0312 → 0.0089); threshold tuning yields 100% sensitivity at 0.31 for screening operating point |
| Bayesian hyperparameter search | Optuna TPE converges in 5× fewer trials than grid search (45 vs 240) |
| Feature-level explanation | SHAP for stakeholder reviewers |

## Short-answer prompts (drop-in)

- **"When would you reach for Bayesian methods over frequentist?"** → Partial-pooling paragraph: nested cohorts, small per-cell samples, joint uncertainty across covariates, need for direct probability statements to non-technical stakeholders.
- **"How do you decide an experiment shipped?"** → Reliability floor first (α ≥ 0.80 or equivalent), then Bayesian effect + 95% HDI, corroborated by a non-parametric sanity check; only decide when both agree.
- **"How do you make Bayesian outputs legible?"** → 95% HDI plots at the layer stakeholders care about, verbal translation ("we'd bet 19:1 the effect is between X and Y"), and calibration diagnostics for the model producing them.
