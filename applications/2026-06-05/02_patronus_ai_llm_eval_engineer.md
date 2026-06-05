# Patronus AI — LLM Evaluation Engineer

**Date:** 2026-06-05
**Location:** Remote / NYC
**Source:** patronus.ai/careers
**Lead project:** Project 1 — AI Safety Red-Team Evaluation
**Role family:** LLM Evaluation / Eval Infra
**Resume version:** resume_v3_eval.pdf

---

## JD keywords to mirror
- "eval harness" / "evaluation harness"
- "LLM-as-judge"
- "production"
- "circuit breaker" / "async" / "throughput"
- "MLflow" / "model registry"

## Cover letter opener (metric hook)

> I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at **96.8% accuracy
> and 850 samples/hr**, with circuit breakers, async batching, and MLflow
> tracking baked in. Cost per sample landed at **$0.018, a 340x reduction**
> versus human review. The interesting part for Patronus is the stacking
> layer: a 47-feature meta-classifier that reconciles disagreement between
> the three judges and surfaces per-model blind spots via Bayesian
> hierarchical modeling. That maps directly to the eval-tooling problems
> Patronus is solving.

## Resume bullets to surface

- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family
- 340x cost reduction vs. human review at $0.018/sample, Krippendorff's alpha = 0.81

## Supporting projects
- LLM Ensemble Bias Detection — 67,500 ratings, 2.5M tokens, MLflow lineage end-to-end
- Breast Cancer Classifier — productionized FastAPI service, <100ms p95, MLflow model registry

## Submission checklist
- [ ] Resume reordered: Red-Team + Bias projects on top
- [ ] Cover letter opens with 340x / 850 samples-hr / MLflow hook
- [ ] "eval harness", "LLM-as-judge", "MLflow" verbatim in resume
- [ ] Salary expectation: open
- [ ] Work auth: US citizen
