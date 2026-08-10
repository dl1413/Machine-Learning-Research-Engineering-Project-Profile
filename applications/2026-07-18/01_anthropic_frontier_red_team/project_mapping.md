# Project → JD Mapping — Anthropic FRT (Cyber)

## AI Safety Red-Team Evaluation Framework (PRIMARY MATCH)

| Their ask | My evidence |
|-----------|-------------|
| Design adversarial evaluations on frontier LLMs | 12,500-pair eval across 6 harm categories, 8-vector MITRE ATLAS-aligned attack taxonomy, multi-turn escalation identified as highest-risk pathway (31.8%) |
| Build reproducible red-team pipelines | 850 samples/hr production pipeline; MLflow experiment tracking; SHAP audit trails; IEEE 2830-2025 compliant |
| Quantify uncertainty honestly | PyMC hierarchical risk model with 95% HDI across GPT-4o / Claude-3.5 / Llama-3.2 |
| Cross-model comparison | Krippendorff's α = 0.81 inter-rater reliability across the 3-model ensemble |
| Defense evaluation | Dual-filter defense reduces harm rate 21.8% → 4.8% (78% reduction) — quantified, not asserted |
| Cost / scale realism | $0.018/sample vs $6.12 human ⇒ 340× reduction — the story of "how do we run this eval every week not every quarter" |

## LLM Ensemble Textbook Bias Detection (SECONDARY — reliability + partial-pooling story)

| Their ask | My evidence |
|-----------|-------------|
| Report uncertainty at the right level | Publisher-level Bayesian partial pooling, 3/5 publishers with credible bias, 12.3% high-uncertainty passages flagged for expert review |
| Statistical rigor | Friedman χ² = 42.73, p < 0.001; α = 0.84; MCMC R-hat < 1.01 |
| Ensemble reliability | 92% pairwise correlation across GPT-4o / Claude-3.5 / Llama-3.2 — a template for FRT multi-model evals |

## Breast Cancer Classification (TERTIARY — production hygiene)

| Their ask | My evidence |
|-----------|-------------|
| Ship evals into a maintained system | MLflow registry + FastAPI < 100 ms p95 latency, Platt-calibrated probabilities (ECE 0.0089) |
| Communicate to non-ML reviewers | Model cards, SHAP explanations, threshold policy for a chosen operating point |

## Short-answer prompts (drop-in)

- **"What eval have you built end-to-end?"** → AI Safety Red-Team paragraph from `cover_letter.md`.
- **"How do you know your eval measures what you claim?"** → 3-model α = 0.81, Bayesian hierarchical with 95% HDI, dual-filter defense delta (21.8% → 4.8%).
- **"What's a surprising finding you had to defend?"** → Multi-turn escalation as highest-risk vector at 31.8%; robust to bootstrap and Bayesian re-estimation.
