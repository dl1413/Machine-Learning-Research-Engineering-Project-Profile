# Application 1 — Anthropic, Research Engineer / Scientist, Alignment Science

- **Date:** 2026-06-16
- **Location:** San Francisco / New York City (hybrid, 25% in office)
- **Source:** job-boards.greenhouse.io/anthropic/jobs/4009165008
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting:** LLM Bias Detection (Bayesian rigor), Breast Cancer (modeling range)

## Resume bullet order (top of Projects section)

1. **AI Safety Red-Team Evaluation Framework** — Dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy / ROC-AUC 0.9923, with Krippendorff's alpha = 0.81 across raters; 340x cost reduction ($6.12 -> $0.018/sample); IEEE 2830-2025 audit trail.
2. **LLM Ensemble Bias Detection** — PyMC partial-pooling hierarchical model over 67,500 ratings (R-hat < 1.01, 95% HDI per publisher); Friedman chi-squared = 42.73, p < 0.001.
3. **Clinical-Grade Breast Cancer Classification** — 99.12% accuracy / 100% precision / ROC-AUC 0.9987; SHAP explanations, calibrated probabilities (ECE 0.0089).

## Cover letter

Dear Anthropic Alignment Science team,

My most relevant work for Anthropic's mission is an independent AI Safety Red-Team Evaluation framework I built and published in early 2026. It ensembles GPT-4o, Claude-3.5-Sonnet, and Llama-3.2 as red-team judges and trains a stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and ROC-AUC 0.9923 on a 12,500-pair benchmark across 6 harm categories. The pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost reduction versus human annotation — while holding inter-rater reliability at Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals per judge, and shipped the whole thing under IEEE 2830-2025 audit-trail requirements.

The second project I'd point to is an LLM-ensemble bias study (4,500 textbook passages, 2.5M tokens, 67,500 ratings) that found statistically significant publisher bias at Friedman chi-squared = 42.73 (p < 0.001) — only defensible because the same three-model ensemble held Krippendorff's alpha = 0.84 and the PyMC partial-pooling model converged at R-hat < 1.01. That Bayesian-first habit is what I'd bring to alignment-science work where the "is this real?" question gates every claim.

I'm finishing an MS in Applied Statistics at RIT (2026) and available for NYC or hybrid. Portfolio, code, and the three technical reports are at github.com/dl1413 — the red-team eval is probably the fastest way to see how I think about safety evaluation.

Best,
Derek Lankeaux

## JD keywords to echo (paste verbatim)

`alignment`, `frontier model`, `evaluation`, `red-team`, `safety research`, `empirical research`, `Constitutional AI`, `RLHF`, `model behavior`, `harm`, `audit`
