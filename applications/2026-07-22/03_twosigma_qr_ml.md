# Two Sigma — Quantitative Researcher, Machine Learning

- **Location:** New York, NY
- **Lead project:** LLM Ensemble Bias Detection (Bayesian hierarchical modeling)
- **Supporting projects:** Breast Cancer Classification, AI Safety Red-Team
- **JD:** https://careers.twosigma.com/careers/JobDetail/New-York-Ny-United-States-Quantitative-Researcher-Machine-Learning/13766
- **Portfolio:** https://github.com/dl1413/machine-learning-research-engineering-project-profile

---

## Cover Letter

Dear Two Sigma Quantitative Research team,

I'm applying for the Quantitative Researcher, Machine Learning role
because the loop described in the JD — large-scale observational data,
inference under bias from complex data-generating processes, models
that trade off rigor and speed — is exactly what my Applied Statistics
MS has trained me to do, and what my three portfolio projects have put
into practice.

The strongest signal is my **LLM Ensemble Bias Detection** system.
67,500 ratings across 4,500 passages from 5 publishers, judged by
three frontier LLMs at α = 0.84 pairwise-correlated agreement (92%).
The interesting layer sits on top: a **PyMC hierarchical model with
partial pooling** that surfaced statistically credible publisher-level
bias in 3/5 publishers (Friedman χ² = 42.73, p < 0.001), with **MCMC
diagnostics reported (R-hat < 1.01, 95% HDI)** and Spearman correlation
uncovering structural editorial relationships (ρ up to 0.74). Same
methodological muscles you'd use to disentangle a persistent alpha
from a data-generating artifact.

Two supporting projects:

- **Breast Cancer Classification** (99.12% accuracy, 100% precision,
  ROC-AUC 0.9987) — the end-to-end modeling half: 8-algorithm
  benchmark, Bayesian hyperparameter search with Optuna TPE (5×
  fewer trials than grid), Platt calibration (ECE reduced 71.5%),
  threshold policies. It's the classification muscle without the
  overfit smell.
- **AI Safety Red-Team Evaluation** — production LLM pipeline at 340×
  cost reduction with audit-grade α = 0.81. This one is here to
  demonstrate the operational half: circuit breakers, exponential
  backoff, MLflow tracking, 80K+ API calls, a report that survives an
  audit.

I know QR at Two Sigma tilts toward finance-specific priors and
market microstructure that I haven't lived in. What I'd bring on day
one is the statistical vocabulary (Bayesian hierarchies, MCMC
diagnostics, calibration, multiple-testing correction, effect sizes)
and the operating hygiene (SQL, Python, MLflow, reproducible
reporting) to be useful while ramping on the domain. Happy to walk
through the reports.

Best,
Derek Lankeaux
MS Applied Statistics, Rochester Institute of Technology (2026)
LinkedIn: https://linkedin.com/in/derek-lankeaux

---

## One-Page Pitch

The core skill Two Sigma seems to be hiring for on this req is
inference under structural noise — models that stay honest when the
data-generating process is doing something the model wasn't asked
about. The bias-detection project is the closest miniature of that:
five publishers, three graders, a hierarchical partial-pooling model
that quantifies uncertainty at the publisher level and flags
high-uncertainty passages (12.3%) for expert review instead of
pretending the point estimate is the answer. That "quantify what you
don't know" habit — plus the calibration and threshold-policy work
from the clinical project, and the pipeline hygiene from the
red-team framework — is what I'd bring to the desk. The market
context I'd have to earn.

---

## Follow-up plan

- **T+0:** Submit via Two Sigma careers portal.
- **T+3 days:** Look up recent Two Sigma authors on arXiv or the
  Two Sigma blog; if any wrote on Bayesian methods or causal
  inference, brief note referencing a specific paper.
- **T+7 days:** If no movement, LinkedIn note to any RIT MS
  Applied Stats alumni at Two Sigma.
- Interview prep: Bayesian reasoning under time pressure,
  regression identification, bias-variance tradeoff, time series
  (per widely-reported Two Sigma DS/QR loop pattern).
