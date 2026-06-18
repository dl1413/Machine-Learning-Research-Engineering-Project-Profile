# 03 — Spotify · Data Scientist II, Business Analytics (NYC)

- **Date pulled:** 2026-06-15
- **Location:** New York, NY (hybrid, Spotify NYC hub)
- **Source:** Built In NYC (job 8071357)
- **JD link:** https://www.builtinnyc.com/job/data-scientist-ii-business-analytics/8071357
- **Lead project:** Project 2 — LLM Ensemble Bias Detection (Bayesian + experimentation framing)
- **Supporting:** Project 1 (LLM eval rigor), Project 3 (modeling range)
- **Resume version to send:** `resume_v3_ds_product.pdf`
- **Cover letter:** Yes (optional → submit one anyway, per playbook §7)

## JD signals (from public summary)

> "Use data analysis, experimentation, and statistical modeling to assess
> initiatives. Analyze A/B tests and quasi-experiments to measure the
> impact of new features."

Keywords to echo: `A/B testing`, `experimentation`, `quasi-experiments`,
`statistical modeling`, `feature impact`, `SQL`, `stakeholder`.

## Tailored cover-letter opener (paste-ready)

> The project I'd point Spotify to is an LLM-ensemble bias-detection study
> I ran this spring: 4,500 passages, 2.5M tokens, 67,500 ratings from
> GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of 5
> publishers showed statistically significant directional bias (**Friedman
> χ² = 42.73, p < 0.001**) — only holds because the pipeline was built to
> defend it: Krippendorff's α = 0.84 across raters, post-hoc Nemenyi
> pairwise comparisons with Bonferroni correction, and a PyMC partial-
> pooling hierarchical model with R-hat < 1.01 producing 95% HDI credible
> intervals per publisher. That Bayesian-first habit — and the
> A/B-test-style mindset around effect sizes, power, and multiple-testing
> correction — is what I'd bring to Spotify's experimentation work.

## Resume bullets to surface

(from `APPLICATION_SNIPPETS.md` → Project 2 → Data Scientist Bayesian/Causal)

- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000)
- Ran Friedman omnibus test (χ² = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize effects
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible directional claims rather than point estimates
- Designed multi-LLM bias-rating system covering 4,500 passages and 2.5M tokens with full MLflow lineage

## Application checklist

- [x] Lead project surfaced first
- [x] 3+ JD phrases echoed (`A/B testing`, `experimentation`, `statistical modeling`)
- [x] Metric hook in opener (Friedman χ² = 42.73, α = 0.84, R-hat < 1.01)
- [x] SQL + Pandas/Polars listed in skills (already on resume)
- [x] Work-auth: US authorized
- [x] Salary expectation: open, NYC market (Spotify DS II ~$140–180k base)
