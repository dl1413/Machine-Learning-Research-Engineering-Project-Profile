# 05 — Hugging Face | Research Engineer, Evaluation

**Location:** Remote (US-friendly)
**Source:** apply.workable.com/huggingface
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 LLM Bias Detection (eval at scale), P3 (ML rigor)
**Role family:** LLM Eval / Research Engineering

---

## JD keywords to mirror

- "open-source"
- "evaluation benchmark"
- "LLM-as-judge"
- "reproducibility"
- "model behavior"
- "Hugging Face datasets / leaderboards"
- "research engineering"

## Resume reordering

1. **AI Safety Red-Team Evaluation Framework** *(top)*
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer Classification

## Top 4 resume bullets (paste under Project 1)

- Designed an open, **reproducible evaluation benchmark**: 12,500 response pairs across 6 harm categories, 47 engineered features, full MLflow lineage and code released alongside the technical report
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 as **LLM-as-judge** raters into a meta-classifier reaching 96.8% accuracy, ROC-AUC 0.9923, Krippendorff alpha = 0.81
- Quantified judge disagreement with PyMC Bayesian hierarchical model — surfaces **model behavior** blind spots per family at 95% HDI granularity
- Built async eval harness at 850 samples/hr with circuit breakers and exponential backoff; usable as a drop-in **leaderboard** scoring backend

## Supporting bullets

- (Bias Detection) Same ensemble at 67,500 ratings / 2.5M tokens scale; Friedman p < 0.001, R-hat < 1.01
- (Breast Cancer) Reproducible 8-model benchmark, SHAP per prediction, FastAPI + Docker — the publication discipline I'd bring to HF Spaces

## Cover letter (paste-ready)

> Dear Hugging Face Team,
>
> Hugging Face is the place that made it normal to publish the eval
> alongside the model, which is exactly the habit I've spent the last
> year practicing.
>
> The work I'd lead with is an open, **reproducible** AI Safety Red-Team
> Evaluation framework I built and published in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as **LLM-as-judge** raters
> and trains a stacking classifier on 47 harm-signal features, reaching
> 96.8% accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark
> across 6 harm categories. The eval harness runs at **850 samples/hr
> for $0.018/sample (340x cheaper than human annotation)** while holding
> inter-rater reliability at Krippendorff's alpha = 0.81, and a PyMC
> Bayesian hierarchical model surfaces **model-behavior** blind spots
> per family at 95% HDI granularity. Code, report, and dataset structure
> are designed to drop straight into a Hugging Face Space / leaderboard.
>
> Two adjacent projects show the same habit at different scales: a
> multi-LLM bias study (67,500 ratings, 2.5M tokens, Friedman p < 0.001,
> R-hat < 1.01) and a clinical-grade breast-cancer classifier (99.12%
> accuracy, FastAPI <100ms p95). All three are on GitHub (dl1413) with
> reports written for someone to reproduce, not just admire.
>
> Open to remote (US) and happy to travel. MS in Applied Statistics at
> RIT, 2026.
>
> Best,
> Derek Lankeaux

## Checklist

- [ ] P1 at top
- [ ] Phrases echoed: "open-source", "evaluation benchmark", "LLM-as-judge", "reproducibility", "model behavior"
- [ ] Hook: 96.8% / 850 samples/hr / 340x
- [ ] Mention `transformers` + `datasets` + `evaluate` libs in skills
- [ ] Link a Hugging Face Space mock-up if time permits
- [ ] Salary: "open, market for remote research engineering"
- [ ] JD PDF saved
