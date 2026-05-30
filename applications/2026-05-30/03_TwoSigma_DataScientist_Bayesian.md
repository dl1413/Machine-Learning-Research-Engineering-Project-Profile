# Two Sigma — Data Scientist / Quantitative Researcher (Bayesian / Causal)

**Location:** New York, NY (on-site)
**Role family:** Data Scientist (Bayesian / Causal Inference)
**Lead project:** LLM Ensemble Bias Detection (Bayesian hierarchical modeling)
**Supporting:** Breast Cancer Classification (calibration, threshold tuning, rigor), AI Safety Red-Team (multi-judge inference)
**JD link:** https://www.twosigma.com/careers/ (paste exact posting URL on submit)
**Resume version:** Resume_Derek_Lankeaux_v_stats.pdf
**Status:** ready_to_submit

---

## Cover Letter

Dear Two Sigma Hiring Team,

I'm applying for the Data Scientist role because the part of data science I care most about — Bayesian inference with defensible uncertainty quantification — is what Two Sigma is known for taking seriously.

One project I'd point to: a multi-LLM bias-detection study where I processed **67,500 ratings over 4,500 textbook passages (2.5M tokens)** through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble, then fit a PyMC Bayesian hierarchical model with partial pooling across publishers. The interesting part is the discipline: MCMC convergence at **R-hat < 1.01 with ESS > 1000**, 92% pairwise inter-LLM correlation, **Krippendorff's alpha = 0.84**, Friedman omnibus chi-squared = 42.73 (**p < 0.001**), and Nemenyi post-hoc to localize effects to specific publishers. The headline result — 3 of 5 publishers showed credible directional bias — only got published because every step of the inference chain could be audited.

A second project carries the same habit into production ML: a clinical-grade ensemble classifier hitting **99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987**, with Platt-calibrated probabilities (ECE 0.0089, 71.5% reduction) and threshold tuning for context-specific decision policies. Calibration, not just accuracy, is the deliverable.

That same Bayesian-first, calibration-first habit is what I'd want to bring to Two Sigma's research work — particularly anywhere that uncertainty quantification, multi-source ensembling, or hierarchical modeling is the right tool but not the easy one.

Based in / available for NYC. Targeting a 2026 start after my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux

---

## Top Resume Bullets

- Fit PyMC Bayesian hierarchical model with partial pooling across 5 publishers; MCMC R-hat < 1.01, ESS > 1000, 95% HDI intervals per publisher and topic
- Ran Friedman omnibus (chi-squared = 42.73, p < 0.001) plus Nemenyi post-hoc to localize effects; held Krippendorff's alpha = 0.84 across LLM raters
- Platt-calibrated probability outputs on 99.12%-accuracy ensemble, reducing ECE 71.5% (0.0312 to 0.0089) for clinically reliable confidence
- Bootstrap CIs and HDI quantification used throughout — point estimates documented with their uncertainty, not without

## JD Keywords to Echo
Bayesian inference, hierarchical modeling, MCMC, posterior, HDI, causal inference, A/B testing, multiple testing correction, uncertainty quantification, calibration, PyMC, statsmodels, SQL.
