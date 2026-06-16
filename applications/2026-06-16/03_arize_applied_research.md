# Application 3 — Arize AI, Applied Research Scientist (LLM Evaluation)

- **Date:** 2026-06-16
- **Location:** Remote (US)
- **Source:** Arize careers / LinkedIn
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting:** LLM Bias Detection (LLM-as-judge depth)

## Resume bullet order

1. AI Safety Red-Team Evaluation — production eval harness, 850 samples/hr, circuit breakers, exponential backoff, MLflow tracking; 47-feature stacking meta-classifier reaching 96.8% agreement with gold labels on 12,500 pairs.
2. LLM Ensemble Bias Detection — 3-LLM ensemble at production scale (67,500 ratings, 2.5M tokens), 92% pairwise inter-LLM correlation, full MLflow lineage; PyMC hierarchical model for judge-disagreement quantification.
3. Breast Cancer Classification — calibration (Platt, ECE 0.0089), threshold tuning, FastAPI <100ms p95 — "treat latency as a feature" habits.

## Cover letter

Dear Arize team,

I'm interested in Arize because I've spent the last few months building exactly the kind of multi-LLM evaluation infrastructure your product abstracts. My recent AI Safety Red-Team Evaluation framework runs a GPT-4o / Claude-3.5 / Llama-3.2 ensemble across 12,500 response pairs at 96.8% accuracy and 850 samples/hr, with circuit-breakered async API integration and MLflow lineage end-to-end. Cost per sample landed at $0.018 — a 340x reduction versus human review.

The interesting part for Arize is the stacking layer: a 47-feature meta-classifier that reconciles disagreement between the three judges and surfaces per-model blind spots via a PyMC Bayesian hierarchical model (95% HDI per judge). That's the LLM-as-judge problem you ship as a product, and I have working code, posteriors, and reliability metrics (Krippendorff's alpha = 0.81) to show for it.

My LLM bias-detection study ran the same ensemble across 67,500 ratings on 4,500 textbook passages and held inter-rater reliability at alpha = 0.84 — the kind of result that needs the reliability scaffolding to be trusted. I'd love to help build that scaffolding as a product.

Remote-available, NYC-time-zone, finishing my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux

## JD keywords to echo

`LLM observability`, `evaluation`, `LLM-as-judge`, `eval harness`, `tracing`, `hallucination`, `drift`, `MLOps`, `production LLM`
