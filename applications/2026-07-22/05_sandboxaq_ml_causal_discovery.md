# SandboxAQ — Machine Learning Engineer, Causal Discovery

- **Location:** New York, NY
- **Lead project:** LLM Ensemble Bias Detection (Bayesian hierarchical / probabilistic modeling)
- **Supporting projects:** Breast Cancer Classification, AI Safety Red-Team
- **JD:** https://www.builtinnyc.com/job/machine-learning-engineer-causal-discovery/6407211
- **Portfolio:** https://github.com/dl1413/machine-learning-research-engineering-project-profile

---

## Cover Letter

Dear SandboxAQ Causal Discovery team,

I'm applying for the Machine Learning Engineer, Causal Discovery role.
The JD describes probabilistic graphical models, large-scale graph
algorithms, and deep learning for causal discovery — a stack that
overlaps meaningfully with the Bayesian hierarchical modeling and
uncertainty quantification I've been publishing on for the last year.

The direct fit is my **LLM Ensemble Bias Detection** framework. Six
publishers x 4,500 passages x three LLM graders → **67,500 ratings**
processed at production scale. On top of that data I built a **PyMC
hierarchical model with partial pooling** and reported the full MCMC
diagnostics (**R-hat < 1.01, ESS, 95% HDI**). The model surfaced
statistically credible publisher-level bias in 3/5 publishers
(**Friedman χ² = 42.73, p < 0.001**), and I ran a Spearman
correlation matrix that uncovered structural editorial relationships
(ρ up to 0.74) — the closest thing to a causal-discovery move the
project scope allowed. Passage-level bootstrap CIs flagged 12.3% of
passages as high-uncertainty and routed them to expert review, so
the pipeline didn't over-claim on cases where the model was ambivalent.

Two supporting projects fill in what a causal-discovery ML engineer
also has to be good at:

- **Breast Cancer Classification** — end-to-end ensemble modeling
  with Bayesian hyperparameter optimization (Optuna TPE, 5× fewer
  trials than grid), Platt calibration (ECE 0.0089), SHAP,
  threshold policies. The engineering half of "modeling under
  uncertainty."
- **AI Safety Red-Team Evaluation** — 12,500-sample LLM ensemble
  pipeline with α = 0.81 and 340× cost reduction, plus MLflow
  tracking, circuit breakers, and IEEE 2830-2025 model cards. The
  production plumbing half.

I know pure causal discovery has its own vocabulary (DAGs, PC / GES /
LiNGAM, do-calculus, PAG) that I'm still building — the closest my
coursework has come is quasi-experimental analysis + hierarchical
Bayesian models. I'd expect to spend the first month getting fluent
in the algorithm layer while contributing on the probabilistic-model
and calibration side from day one.

Best,
Derek Lankeaux
MS Applied Statistics, Rochester Institute of Technology (2026)
LinkedIn: https://linkedin.com/in/derek-lankeaux

---

## One-Page Pitch

Causal discovery on real observational data is a probabilistic-model
problem with graph-algorithm scaffolding. My bias-detection project
lives in the same neighborhood: partial-pooling hierarchy, MCMC
diagnostics, Spearman-based structural correlation across publishers.
The calibration + threshold-policy work from the clinical project
carries over — a discovered edge is only useful if the associated
probability is honest. And the red-team framework covers the
last-mile hygiene: 80K+ API calls with MLflow tracking, circuit
breakers, backoff, and publication-grade reports. What I'd need from
the team on day one is the causal-inference-specific algorithm
vocabulary and the domain priors (SandboxAQ's markets are pharma,
security, financial-services quantum — none of which I've worked in).
That's a ramp, and I plan for it.

---

## Follow-up plan

- **T+0:** Submit via SandboxAQ / Built In NYC portal.
- **T+3 days:** Read 2-3 recent Causal Discovery publications from
  the SandboxAQ team (if any are on arXiv); reach out to a first
  author on LinkedIn with a specific question tied to my hierarchical
  bias project.
- **T+7 days:** If no movement, look for RIT / Applied Stats alumni
  at SandboxAQ.
- Interview prep: PC / GES / LiNGAM algorithms, do-calculus
  fundamentals, DoWhy / CausalNex libraries, probabilistic graphical
  models (Koller & Friedman chapters most relevant to the JD).
