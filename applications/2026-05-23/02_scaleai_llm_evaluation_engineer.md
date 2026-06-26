# 02 — Scale AI | LLM Evaluation Engineer

**Location:** NYC / Remote
**Source:** scale.com/careers
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 — LLM Bias Detection (eval at scale), P3 (model selection rigor)
**Role family:** LLM Eval Engineering / Eval Infra

---

## JD keywords to mirror

- "evaluation harness"
- "LLM-as-judge"
- "inter-rater reliability"
- "throughput"
- "RLHF data quality"
- "eval pipeline at scale"

## Resume reordering

1. **AI Safety Red-Team Evaluation Framework** *(top)*
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer Classification

## Top 4 resume bullets (paste under Project 1)

- Built production **evaluation harness** processing **850 samples/hr** with circuit breakers, exponential backoff, async batching, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching **96.8% agreement with gold human labels** on 12,500 pairs across 6 harm categories
- Held **inter-rater reliability** at Krippendorff's alpha = 0.81; quantified judge disagreement via PyMC Bayesian hierarchy with 95% HDI per model family
- Cut eval cost 340x ($6.12 → $0.018/sample), making continuous eval feasible for **RLHF data quality** loops at production throughput

## Supporting bullets (LLM Bias Detection)

- Operated the same 3-LLM ensemble at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage; alpha = 0.84, 92% pairwise correlation

## Cover letter (paste-ready)

> Dear Scale AI Team,
>
> Scale's eval infrastructure is what most labs end up rebuilding badly in
> house, so the LLM Evaluation Engineer role is one I'd take seriously.
>
> I recently shipped a 3-model **LLM-as-judge** evaluation harness —
> GPT-4o, Claude-3.5, Llama-3.2 — that auto-grades 12,500 response pairs at
> 96.8% accuracy and **850 samples/hr**, with circuit breakers, async
> batching, and MLflow tracking baked in. Cost per sample landed at
> **$0.018, a 340x reduction** versus human review. The interesting part
> for Scale is the stacking layer: a 47-feature meta-classifier that
> reconciles disagreement between the three judges and surfaces per-model
> blind spots via Bayesian hierarchical modeling — exactly the kind of
> work that sits between raw eval throughput and **RLHF data quality**.
>
> I've run the same ensemble at a different scale: 67,500 LLM ratings
> over 2.5M tokens of textbook content, holding **inter-rater
> reliability** at Krippendorff's alpha = 0.84 and surfacing publisher-
> level bias at p < 0.001 (Friedman). And a clinical-grade breast-cancer
> classifier (99.12% accuracy, 100% precision, <100ms p95 FastAPI) is
> where I learned to ship the eval *and* the production service together.
>
> I'm based in / available for NYC and open to remote, targeting a 2026
> start after my MS in Applied Statistics at RIT. Portfolio and reports
> are on GitHub (dl1413).
>
> Best,
> Derek Lankeaux

## Checklist

- [ ] P1 at top of Projects section
- [ ] Phrases echoed: "evaluation harness", "LLM-as-judge", "inter-rater reliability", "RLHF data quality"
- [ ] Hook: 96.8% / 850 samples/hr / 340x
- [ ] Skills include: MLflow, PyMC, XGBoost, FastAPI, Docker, async API
- [ ] Salary: "open"
- [ ] JD PDF saved
