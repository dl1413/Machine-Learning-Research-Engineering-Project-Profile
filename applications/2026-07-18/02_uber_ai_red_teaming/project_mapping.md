# Project → JD Mapping — Uber AI Red Teaming & Model Risk

## AI Safety Red-Team Framework (PRIMARY MATCH)

| Their ask | My evidence |
|-----------|-------------|
| Reusable eval pipelines for continuous red teaming | 850 samples/hr production pipeline, MLflow-tracked, 6 harm categories × 8 attack vectors, plug-and-play across new prompt sets |
| Direct AI red-teaming experience | 8-category MITRE ATLAS-aligned taxonomy; multi-turn escalation identified as highest-risk vector at 31.8% |
| Measure defense effectiveness | Dual-filter defense reduces harm rate 21.8% → 4.8% (78% reduction) — measured, not asserted |
| Cross-model reliability | Krippendorff's α = 0.81 across GPT-4o / Claude-3.5 / Llama-3.2 |
| Governance / audit posture | SHAP audit trails, model cards, IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act compliance |
| Uncertainty for risk decisions | Bayesian hierarchical risk model with 95% HDI per model per harm category |

## LLM Ensemble Textbook Bias Detection (SECONDARY — pattern reuse)

| Their ask | My evidence |
|-----------|-------------|
| Extend eval pattern to a new modality | Same 3-model ensemble structure, retargeted to 4,500 passages / 67,500 ratings — proves the framework generalizes |
| Detect subtle model-behavior differences | Friedman χ² = 42.73, p < 0.001; 3/5 publishers with credible bias under partial pooling |
| Ship at scale | 2.5M tokens processed with circuit breakers, exponential backoff, MLflow tracking |

## Breast Cancer Classification (TERTIARY — production + calibration)

| Their ask | My evidence |
|-----------|-------------|
| Ship models with calibrated confidence | Platt scaling reduces ECE 71.5% (0.0312 → 0.0089); threshold policy tuned per operating point |
| Serve at latency | FastAPI < 100 ms p95, MLflow model registry |
| Explain to non-technical reviewers | SHAP-based feature importance, model cards for clinical / regulatory readers |

## Short-answer prompts (drop-in)

- **"Describe a red-team eval you built."** → AI Safety Red-Team paragraph.
- **"How do you know the model got safer?"** → 21.8% → 4.8% harm-rate delta with the dual-filter defense; α = 0.81 IRR keeps the label side honest.
- **"How do you communicate risk without over-claiming?"** → 95% HDI intervals per model per category; passage-level bootstrap CIs on the bias project flag 12.3% for expert review rather than auto-decide.
