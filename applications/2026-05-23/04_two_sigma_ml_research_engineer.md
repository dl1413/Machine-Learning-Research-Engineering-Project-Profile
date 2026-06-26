# 04 — Two Sigma | ML Research Engineer (Inference / Bayesian)

**Location:** New York City (on-site)
**Source:** twosigma.com/careers
**Lead project:** P2 — LLM Ensemble Bias Detection (Bayesian heavy)
**Supporting:** P1 Red-Team (eval at scale), P3 (model selection rigor)
**Role family:** DS — Bayesian / Causal Inference

---

## JD keywords to mirror

- "Bayesian"
- "hierarchical model"
- "MCMC"
- "uncertainty quantification"
- "causal inference"
- "hypothesis testing"
- "experimental design"

## Resume reordering

1. **LLM Ensemble Textbook Bias Detection** *(top — lead with stats rigor)*
2. AI Safety Red-Team Evaluation Framework
3. Breast Cancer Classification

## Top 4 resume bullets (paste under Project 2)

- Fit PyMC **Bayesian hierarchical model** with partial pooling across publishers on 67,500 LLM ratings over 4,500 passages and 2.5M tokens; achieved **MCMC** convergence (R-hat < 1.01, ESS > 1000)
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible "this publisher is biased" claims rather than point estimates — directly applicable to **uncertainty quantification** on noisy signals
- Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons; surfaced significant directional bias in 3 of 5 publishers
- Held inter-rater reliability at Krippendorff's alpha = 0.84 / 92% pairwise correlation across GPT-4o, Claude-3.5, Llama-3.2 — i.e., the input distribution is trustworthy before any downstream inference

## Supporting bullets

- (Red-Team Eval) Same Bayesian-first habit at LLM safety scale: 95% HDI risk intervals across judges on 12,500 evals
- (Breast Cancer) Nested cross-validation, calibrated probabilities, SHAP attribution — the model-selection discipline that translates to alpha research

## Cover letter (paste-ready)

> Dear Two Sigma Research Team,
>
> Two Sigma is one of the few places where Bayesian discipline isn't a
> nice-to-have, so I want to lead with the project where I leaned on it
> hardest: an LLM-ensemble bias-detection study I ran last quarter.
> 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings from GPT-4o /
> Claude-3.5 / Llama-3.2. The headline finding — that 3 of 5 publishers
> showed statistically significant directional bias (**Friedman
> chi-squared = 42.73, p < 0.001**) — only holds because the pipeline was
> built to defend it: Krippendorff's alpha = 0.84 across raters, 92%
> pairwise correlation, and a **PyMC partial-pooling hierarchical model**
> with R-hat < 1.01 producing 95% HDI credible intervals per publisher.
>
> Two adjacent projects reinforce the same toolkit. An AI Safety Red-Team
> Evaluation framework (96.8% accuracy across 12,500 pairs, Bayesian
> hierarchical posteriors over judges, 340x cost reduction) and a clinical-
> grade breast-cancer classifier (99.12% accuracy, 100% precision, ROC-AUC
> 0.9987, <100ms p95) — different domains, same workflow of frame the
> question, pick the prior, fit the model, defend the interval, ship the
> service.
>
> I'm based in NYC and excited about an on-site research engineering role.
> MS in Applied Statistics from RIT in 2026 — Bayesian Inference, MCMC,
> Experimental Design were the specialization. Portfolio and reports are
> on GitHub (dl1413).
>
> Best,
> Derek Lankeaux

## Checklist

- [ ] P2 at top of Projects section
- [ ] Phrases echoed: "Bayesian", "hierarchical model", "MCMC", "uncertainty quantification", "hypothesis testing"
- [ ] Hook: Friedman 42.73 / p<0.001 / R-hat < 1.01
- [ ] Skills include PyMC, ArviZ, NumPyro, Stan
- [ ] Two Sigma typically requires on-site — confirm in app
- [ ] Salary: "open, market for NYC research engineering"
- [ ] JD PDF saved
