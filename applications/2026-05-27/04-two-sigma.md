# 04 — Two Sigma, Quantitative Researcher (ML & Statistics)

- **Location:** New York City (on-site)
- **Source:** twosigma.com/careers (direct)
- **Role family:** Data Scientist (Bayesian / Causal)
- **Lead project:** LLM Ensemble Textbook Bias Detection (Bayesian rigor)
- **Supporting projects:** Breast Cancer (modeling discipline), Red-Team (eval infra)
- **Resume version:** `resume_v3_ds.pdf`

## Cover letter (paste-ready)

Hi Two Sigma team,

One project I'd point to is an LLM-ensemble bias-detection study I ran last
quarter: 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings from
GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of 5
publishers showed statistically significant directional bias (Friedman
chi-squared = 42.73, p < 0.001, post-hoc Nemenyi to localize) — only holds
because the pipeline was built to defend it: Krippendorff's alpha = 0.84
across raters, 92% pairwise correlation, and a PyMC partial-pooling
hierarchical model with R-hat < 1.01 and ESS > 1000, producing 95% HDI
credible intervals per publisher rather than point estimates. That
Bayesian-first habit — treating uncertainty as the deliverable, not an
afterthought — is what I'd want to bring to Two Sigma's research work.

Two other projects fill out my profile: a clinical-grade ensemble
classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987) built by
benchmarking 8 algorithms under nested cross-validation, and an LLM
red-team eval harness running at 850 samples/hr with full MLflow lineage.

I'm finishing my MS in Applied Statistics at RIT (specialization: Bayesian
methods, experimental design) and would love an NYC on-site role starting
2026.

Best,
Derek Lankeaux
LinkedIn: linkedin.com/in/derek-lankeaux | GitHub: github.com/dl1413

## Resume top-3 bullets

1. Fit PyMC Bayesian hierarchical model with partial pooling across publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000) and 95% HDI credible intervals
2. Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize publisher-level effects across 67,500 ratings
3. Designed multi-LLM rating pipeline covering 4,500 passages and 2.5M tokens; inter-rater reliability held at Krippendorff's alpha = 0.84, 92% pairwise correlation

## JD keyword targets

- [ ] "Bayesian"
- [ ] "hierarchical" / "MCMC"
- [ ] "causal inference" / "experimental design"
- [ ] "statistical rigor"
- [ ] "hypothesis testing"
