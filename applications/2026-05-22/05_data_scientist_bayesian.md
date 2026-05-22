# Data Scientist — Experimentation / Causal Inference (NYC + Remote)

- **Location:** NYC / US-Remote (Built In NYC + Wellfound listings; target product
  DS roles citing experimentation, causal inference, Bayesian)
- **Lead project:** P2 — LLM Ensemble Textbook Bias Detection
- **Supporting:** P3 (modeling rigor), P1 (eval)
- **Fit note:** Strong. Use for any DS req naming Bayesian / MCMC / hierarchical /
  causal inference / A/B testing. Swap the company name + 3 JD phrases per posting.
- **JD phrases to echo:** experimentation, A/B testing, causal inference, Bayesian
  statistics, hypothesis testing, credible intervals, production models.

## Resume — Projects section order
1. LLM Ensemble Textbook Bias Detection (lead)
2. Clinical-Grade Breast Cancer ML Classification
3. AI Safety Red-Team Evaluation

### Top resume bullets
- Fit PyMC **Bayesian** hierarchical model with partial pooling across publishers; MCMC convergence (R-hat < 1.01, ESS > 1000), 95% **credible intervals** (HDI) per group
- Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) + post-hoc Nemenyi pairwise comparisons to localize effects — defensible **hypothesis testing**, not point estimates
- Designed the measurement instrument: 67,500 ratings, 2.5M tokens, Krippendorff's alpha = 0.84, 92% pairwise correlation across raters
- Track record shipping **production models** (FastAPI, MLflow, <100ms p95) from the breast-cancer project

### Skills to surface
Bayesian/PyMC/ArviZ, MCMC diagnostics, A/B testing & power analysis, causal inference, multiple-testing correction, SQL, Python/R.

## Cover letter
> Dear [Company] Data Science team,
>
> One project I'd point to is an LLM-ensemble bias-detection study: 4,500 passages,
> 2.5M tokens, 67,500 ratings from GPT-4o / Claude-3.5 / Llama-3.2. The headline —
> that 3 of 5 publishers showed statistically significant directional bias (Friedman
> chi-squared = 42.73, p < 0.001) — only holds because the pipeline was built to
> defend it: Krippendorff's alpha = 0.84 across raters, 92% pairwise correlation, and
> a PyMC partial-pooling **Bayesian** hierarchical model with R-hat < 1.01 producing
> 95% **credible intervals** per publisher. That Bayesian-first habit, plus careful
> **hypothesis testing** and multiple-testing correction, is what I'd bring to
> [Company]'s **experimentation** and **causal inference** work.
>
> I also ship: a separate project took an 8-algorithm benchmark to a **production
> model** behind FastAPI (<100ms p95, MLflow registry), so I'm comfortable owning the
> path from question → experiment → SQL → model → uncertainty → stakeholder readout.
>
> I'm finishing an MS in Applied Statistics at RIT (2026), available NYC or remote,
> US work-authorized. Portfolio and three technical reports on GitHub (dl1413).
> Salary open, targeting market for the role/location.
>
> Best, Derek Lankeaux · dlankeaux12@gmail.com · linkedin.com/in/derek-lankeaux
