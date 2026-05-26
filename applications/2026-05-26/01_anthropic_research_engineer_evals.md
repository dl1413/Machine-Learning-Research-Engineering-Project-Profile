# 01 — Anthropic · Research Engineer, Evaluations

- **Location:** Remote (US) / NYC presence
- **Family:** AI Safety / LLM Eval
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 Bias Detection (ensemble + Bayesian), P3 Breast Cancer (rigor)
- **JD link:** https://www.anthropic.com/jobs (Research Engineer, Evaluations)
- **Resume version:** `resume_v3_safety.pdf`
- **Cover letter:** YES

## JD keywords to echo verbatim (≥3)
1. "model evaluations" / "evals"
2. "harm" / "harmful behaviors"
3. "Constitutional AI"
4. "red-team" / "red-teaming"
5. "LLM-as-judge"

## Resume bullet stack (top of Projects section)

**AI Safety Red-Team Evaluation Framework (lead)**
- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while maintaining Krippendorff's α = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

**LLM Textbook Bias Detection (supporting)**
- Operated 3-LLM ensemble at production scale: 67,500 ratings over 4,500 passages / 2.5M tokens; Friedman χ² = 42.73, p < 0.001
- Bayesian hierarchical model (PyMC, partial pooling, R-hat < 1.01) producing 95% HDI per publisher

**Breast Cancer Classification (supporting — rigor)**
- 8-algorithm benchmark, 99.12% accuracy, 100% precision, FastAPI service under 100ms p95

## Cover letter

> Dear Anthropic Evaluations team,
>
> My most relevant work for Anthropic's mission is an independent AI Safety
> Red-Team Evaluation framework I built and published in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
> stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and
> ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories.
> The pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost
> reduction versus human annotation — while holding inter-rater reliability
> at Krippendorff's α = 0.81. I paired it with a PyMC Bayesian hierarchical
> model that produces 95% HDI risk intervals per judge, and shipped the
> whole thing under IEEE 2830-2025 audit-trail requirements. The framing
> drew explicitly on Constitutional AI as the spec for what "harmful" means
> in the rubric.
>
> Two adjacent projects show the same habits at different scales: an LLM
> bias-detection study (67,500 ratings, Krippendorff α = 0.84, Friedman
> χ² = 42.73, p < 0.001 on 3/5 publishers) and a clinical-grade classifier
> (99.12% accuracy, zero false positives, <100ms p95 FastAPI). I'd love to
> bring that combination of eval rigor and production throughput to
> Anthropic's evaluations team — particularly red-team eval infrastructure
> and the meta-classifier work for reconciling multi-judge disagreement.
>
> I'm available for New York City and open to remote, targeting a 2026
> start once I wrap my MS in Applied Statistics at RIT. Portfolio, code,
> and the three technical reports are on GitHub (dl1413).
>
> Best,
> Derek Lankeaux
