# 03 — Two Sigma · Quantitative Researcher (Modeling)

| Field | Value |
|---|---|
| **Company** | Two Sigma |
| **Role** | Quantitative Researcher — Modeling (or "Statistical Modeling Researcher") |
| **Location** | NYC (on-site, hybrid) |
| **Source** | careers.twosigma.com (apply direct) |
| **JD link** | _VERIFY before submit:_ https://www.twosigma.com/careers/ (filter: Researcher / Quant / Bayesian) |
| **Lead project** | P2 — LLM Bias Detection (Bayesian hierarchical modeling) |
| **Supporting** | P3 — Breast Cancer (rigorous model selection, calibration); P1 — Red-Team (ensemble + uncertainty) |
| **Resume version** | `resume_v3_stats.pdf` |
| **Cover letter** | Yes — see below |
| **Referral** | RIT alumni network — check for Two Sigma researchers |

## JD keyword echo (fill in after reading JD)

- [ ] Bayesian / hierarchical / MCMC
- [ ] uncertainty quantification / credible intervals
- [ ] feature engineering / cross-validation
- [ ] [Two Sigma–specific: "research platform", "out-of-sample", "alpha"]
- [ ] [Two Sigma–specific phrase 2]

## Resume bullets (top 4 — paste into "Selected Projects")

From `APPLICATION_SNIPPETS.md` → P2 → "Data Scientist (Bayesian / Causal)":

- Fit PyMC Bayesian hierarchical model with partial pooling across 5 publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000)
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible directional bias claims rather than point estimates
- Ran Friedman omnibus test (χ² = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize effects
- Validated 3-rater ensemble at Krippendorff's α = 0.84 and 92% pairwise correlation across GPT-4o, Claude-3.5, Llama-3.2 over 67,500 ratings

## Cover letter draft

Dear Two Sigma research team,

One project I'd point to is an LLM-ensemble bias-detection study I
shipped this year: 4,500 textbook passages, 2.5M tokens, 67,500 LLM
ratings from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding —
that 3 of 5 publishers showed statistically significant directional bias
(Friedman χ² = 42.73, p < 0.001) — only holds because the pipeline
was built to defend it: Krippendorff's α = 0.84 across raters, 92%
pairwise correlation, and a PyMC partial-pooling hierarchical model with
R-hat < 1.01 producing 95% HDI credible intervals per publisher. That
Bayesian-first habit — defending each claim with a posterior, not a
point estimate — is what I want to bring to Two Sigma's modeling work.

The other two projects round out my profile. A clinical classifier at
99.12% accuracy / 100% precision / ROC-AUC 0.9987 (8-algorithm
benchmark with nested CV, Platt-calibrated to ECE 0.0089) shows the
same rigor on tabular signal. And a 3-LLM red-team evaluation
framework (96.8% accuracy, 340x cost reduction vs. human annotation)
shows I can ship production eval infra end-to-end.

I'm finishing an MS in Applied Statistics at RIT (Bayesian methods,
experimental design, causal inference) and based in / available for
NYC. Portfolio, code, and the three technical reports are on my
GitHub (dl1413). Happy to walk through any of them.

Best,
Derek Lankeaux
