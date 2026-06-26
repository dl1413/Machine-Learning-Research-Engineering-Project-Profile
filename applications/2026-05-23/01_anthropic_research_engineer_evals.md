# 01 — Anthropic | Research Engineer, Alignment Evals

**Location:** Remote (US) / NYC presence
**Source:** anthropic.com/jobs
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 (multi-LLM ensemble rigor), P3 (clinical-grade benchmarking discipline)
**Role family:** AI Safety / Eval Engineering

---

## JD keywords to mirror (paste verbatim into resume + letter)

- "red-teaming"
- "model evaluation"
- "Constitutional AI"
- "LLM-as-judge"
- "harm classification"
- "production eval harness"
- "audit trail"

## Resume reordering (Projects section, top → bottom)

1. **AI Safety Red-Team Evaluation Framework** *(promote to top)*
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer Classification

## Top 4 resume bullets (paste under Project 1)

- Engineered dual-stage **LLM-as-judge** ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 **harm classification** categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost **340x** ($6.12 → $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across the three judges
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals to support **Constitutional AI**-style training data review
- Shipped IEEE 2830-2025-compliant **production eval harness** processing 850 samples/hr with SHAP explanations, MLflow lineage, and full **audit trail** provenance

## Cover letter (paste-ready)

> Dear Anthropic Alignment Team,
>
> Anthropic is the lab whose published work I refer to most — the
> Constitutional AI and sycophancy papers in particular shaped how I built
> my own red-team eval pipeline — so the Research Engineer, Alignment Evals
> role is the one I most want to do well.
>
> My most relevant work for this team is an independent AI Safety Red-Team
> Evaluation framework I built and published in January 2026. It ensembles
> GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
> stacking classifier on 47 harm-signal features, reaching **96.8% accuracy
> and ROC-AUC 0.9923** against a 12,500-pair benchmark across 6 harm
> categories. The harness runs at **850 samples/hr for $0.018/sample — a
> 340x cost reduction** versus human annotation — while holding inter-rater
> reliability at Krippendorff's alpha = 0.81. I paired it with a PyMC
> Bayesian hierarchical model producing 95% HDI risk intervals per judge,
> and shipped the whole thing under IEEE 2830-2025 audit-trail requirements.
>
> Two other projects pressure-tested the same habits at different scales:
> a multi-LLM bias-detection study (67,500 ratings, Friedman chi-squared =
> 42.73, p < 0.001, R-hat < 1.01 across publishers) and a clinical-grade
> breast-cancer classifier (99.12% accuracy, zero false positives, SHAP
> per prediction, <100ms p95 FastAPI). Together they're how I practice
> the loop of "build the eval, defend it statistically, ship it as
> infrastructure."
>
> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
> code, and the three technical reports are on my GitHub (dl1413). Happy
> to walk through the red-team eval — probably the fastest way to see how
> I think about Anthropic's evaluation problems.
>
> Best,
> Derek Lankeaux

## Checklist before submit

- [ ] Resume bullet order rearranged (P1 at top)
- [ ] 3+ JD phrases in resume + letter ("red-teaming", "Constitutional AI", "LLM-as-judge", "audit trail")
- [ ] Cover letter opens with 340x / 96.8% / Krippendorff 0.81 hook
- [ ] LinkedIn, GitHub, Portfolio URLs live
- [ ] PyTorch in skills (Anthropic uses PyTorch internally)
- [ ] Salary expectation: "open, targeting market for the role/location"
- [ ] Save JD PDF to this folder before submit
