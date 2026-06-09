# Two Sigma — Data Scientist, Modeling Research

**Location:** New York, NY · **Lead project:** Project 2 (LLM Bias / Bayesian Hierarchical)
**Source:** twosigma.com/careers · **Snippet base:** APPLICATION_SNIPPETS §2 (Bayesian / Causal)

## JD keyword echo (≥3 verbatim)
- "Bayesian", "hierarchical model", "uncertainty quantification", "hypothesis testing", "production pipeline"

## Resume Projects-section order
1. **LLM Ensemble Bias Detection** (LEAD — Bayesian hierarchical core)
2. Breast Cancer ML Classification (modeling rigor / calibration)
3. AI Safety Red-Team Evaluation (production scale)

## Tailored resume bullets
- Fit **PyMC Bayesian hierarchical model** with partial pooling across publishers and topics; MCMC convergence at **R-hat < 1.01, ESS > 1000**
- Produced **95% HDI credible intervals** per publisher / per topic enabling defensible directional claims over 67,500 LLM-judge ratings
- Ran Friedman omnibus test (**chi-squared = 42.73, p < 0.001**) + Nemenyi post-hoc pairwise comparisons; corrected with Bonferroni / FDR where appropriate
- Held inter-rater reliability at **Krippendorff's alpha = 0.84** and 92% pairwise correlation across 3 LLM raters; flagged 12.3% high-uncertainty passages via bootstrap CIs for expert review
- Platt-calibrated downstream classifier (separate project) — **ECE reduced 71.5%** (0.0312 -> 0.0089) — demonstrating discipline on probability-quality, not just point accuracy

## Cover letter

> Dear Two Sigma Modeling Research,
>
> One project I'd point to is an LLM-ensemble bias-detection study I ran
> this spring: 4,500 textbook passages, 2.5M tokens, **67,500 LLM ratings**
> from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of
> 5 publishers showed statistically significant directional bias (**Friedman
> chi-squared = 42.73, p < 0.001**) — only holds because the pipeline was
> built to defend it: **Krippendorff's alpha = 0.84** across raters, 92%
> pairwise correlation, and a **PyMC partial-pooling hierarchical model**
> with R-hat < 1.01 producing 95% HDI credible intervals per publisher.
> Bootstrap CIs flagged 12.3% of passages as high-uncertainty before any
> claim was made. That Bayesian-first habit — quantify uncertainty before
> you quote a number — is what I'd want to bring to your modeling work.
>
> Two related projects show the same discipline transferred elsewhere: a
> red-team eval pipeline (96.8% accuracy, 12,500 pairs, 850 samples/hr in
> production) and a clinical classifier where I cared as much about
> Platt calibration (**ECE -71.5%**) and threshold tuning as I did about
> the headline 99.12% accuracy. I default to running uncertainty
> quantification, multiple-testing correction, and power analysis before
> shipping a result.
>
> Finishing my MS in Applied Statistics at RIT (2026), NYC-based, US
> work-authorized.
>
> Best,
> Derek Lankeaux
