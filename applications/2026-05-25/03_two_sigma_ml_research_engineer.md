# Application 3 — Two Sigma · Machine Learning Research Engineer

| Field | Value |
|---|---|
| Date | 2026-05-25 (Mon) |
| Company | Two Sigma |
| Role | Machine Learning Research Engineer |
| Role family | NYC Finance ML / Applied Research |
| Location | NYC (on-site) |
| Source | twosigma.com/careers |
| Lead project | Project 3 — Clinical-Grade Breast Cancer ML (model selection rigor + production deploy) |
| Supporting | Project 1 — Red-Team Eval (eval infra at scale), Project 2 — Bayesian hierarchical bias study |
| Resume version | resume_v3_mle.pdf |
| Cover letter | Yes |
| Referral | No |

---

## Tailored resume bullets

**Clinical-Grade ML Classification System** — *Independent Research, Jan 2026*
- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting + 2) under nested cross-validation; shipped winning ensemble at 99.12% accuracy / ROC-AUC 0.9987
- Applied VIF-based multicollinearity pruning, SMOTE class-balancing, and RFE feature selection to lift recall to 98.59% while holding precision at 100%
- Productionized winner behind FastAPI + MLflow model registry; **<100ms p95 latency**, SHAP-per-prediction explanations

**LLM Ensemble Bias Detection** — *Independent Research, Jan 2026*
- Fit PyMC Bayesian hierarchical model with partial pooling across 5 publishers; MCMC convergence at R-hat < 1.01, ESS > 1000
- 95% HDI credible intervals per publisher + Friedman omnibus (chi-squared = 42.73, p < 0.001) with post-hoc Nemenyi pairwise comparisons

**AI Safety Red-Team Evaluation** — *Independent Research, Jan 2026*
- Built eval harness at 850 samples/hr with circuit breakers, exponential backoff, MLflow lineage; 96.8% accuracy / ROC-AUC 0.9923 on 12,500-pair benchmark

## Cover letter

Dear Two Sigma Hiring Team,

I'm applying for the ML Research Engineer role because the way Two Sigma
treats modeling — pick by evidence, ship the full pipeline, treat latency as
a feature — matches how I work. A representative recent project: I
benchmarked 8 algorithms (Random Forest through stacking ensembles) under
nested cross-validation on a clinical-grade classification task and landed at
**99.12% accuracy, 100% precision, ROC-AUC 0.9987**, then productionized the
winner behind a FastAPI + MLflow service at **<100ms p95 latency** with SHAP
explanations per prediction.

Two other projects from this year speak to the research side of the role. A
PyMC Bayesian hierarchical study on textbook bias (4,500 passages, 67,500
LLM ratings, MCMC R-hat < 1.01) produced 95% HDI credible intervals per
publisher and a defensible Friedman omnibus at p < 0.001 — the kind of
statistical care I'd want on any signal that goes near a trading decision.
And an AI Safety Red-Team eval harness running at 850 samples/hr with
circuit breakers and async batching shows I can build production data
infrastructure, not just notebooks.

I'm finishing my MS in Applied Statistics at RIT (2026, specialization in
Bayesian methods and experimental design), based in NYC, and ready to start
on-site. Would love to talk.

Best,
Derek Lankeaux

## JD keyword echoes

- "machine learning research"
- "production model" / "production ML pipeline"
- "feature engineering"
- "Bayesian"
- "cross-validation"
- "low-latency"

## Quality checklist

- [x] Lead project re-ordered (Breast Cancer top)
- [x] Metric hook (99.12% / <100ms p95)
- [x] 3+ JD phrases verbatim
- [x] NYC on-site availability stated
- [x] Salary: "open, targeting market for NYC"
- [x] Work auth: US authorized
