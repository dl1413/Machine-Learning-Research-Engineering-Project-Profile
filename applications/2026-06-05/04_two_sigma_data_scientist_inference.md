# Two Sigma — Data Scientist, Inference & Bayesian Methods

**Date:** 2026-06-05
**Location:** NYC (hybrid)
**Source:** twosigma.com/careers
**Lead project:** Project 2 — LLM Ensemble Bias Detection (Bayesian angle)
**Role family:** Data Scientist (Bayesian / Causal)
**Resume version:** resume_v3_ds_bayesian.pdf

---

## JD keywords to mirror
- "Bayesian" / "hierarchical model"
- "MCMC" / "posterior"
- "hypothesis testing"
- "experimental design"
- "uncertainty quantification"

## Cover letter opener (metric hook)

> One project I'd point to is an LLM-ensemble bias-detection study I ran
> last quarter: **4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings**
> from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of
> 5 publishers showed statistically significant directional bias
> (**Friedman χ² = 42.73, p < 0.001**) — only holds because the pipeline
> was built to defend it: Krippendorff's alpha = 0.84 across raters, 92%
> pairwise correlation, and a **PyMC partial-pooling hierarchical model
> with R-hat < 1.01** producing 95% HDI credible intervals per publisher.
> That Bayesian-first habit is what I'd want to bring to Two Sigma's
> inference work.

## Resume bullets to surface

- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000)
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible inference rather than point estimates
- Ran Friedman omnibus test (χ² = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize effects
- Operated 3-LLM ensemble at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage

## Supporting projects
- AI Safety Red-Team Eval — Bayesian 95% HDI risk intervals per judge
- Breast Cancer Classifier — 8-algorithm benchmark, nested CV, calibration

## Submission checklist
- [ ] Resume reordered: Bias project + Bayesian skills near top
- [ ] Cover letter opens with χ² / p < 0.001 / R-hat hook
- [ ] "Bayesian", "MCMC", "hierarchical" verbatim in resume Skills
- [ ] PyMC, ArviZ, NumPyro called out
- [ ] Salary expectation: open, targeting NYC market
