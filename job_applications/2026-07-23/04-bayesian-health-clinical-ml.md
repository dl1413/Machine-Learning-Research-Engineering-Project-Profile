# Bayesian Health — Data Scientist / ML Engineer, Clinical ML

**Location:** Remote · **Apply:** https://bayesianhealth.com/careers · **Fit:** ★★★★★

---

Dear Bayesian Health team,

I'm applying for a Data Scientist / ML role. The name of the company plus a
healthcare mission is a near-perfect overlap with my MS focus (Applied Statistics,
Bayesian Methods) and my most decision-critical published project.

My **Clinical-Grade Breast Cancer Classification System** exceeded human expert
performance (90-95% baseline) at **99.12% accuracy, 100% precision (zero false
positives), 98.59% recall, ROC-AUC 0.9987**, benchmarked across 8 algorithms
(RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) with Optuna TPE converging in
**5× fewer trials than grid search (45 vs 240)**. Beyond raw performance, the parts
you actually need to deploy in a clinical setting: **Platt calibration reduced ECE
by 71.5% (0.0312 → 0.0089)**, context-adaptive thresholds (0.31 for 100% screening
sensitivity), VIF multicollinearity analysis, SMOTE balancing, RFE selection, SHAP
for clinical transparency, IEEE 2830-2025 fairness auditing, MLflow registry, and
FastAPI serving under **100ms p95 latency**.

The Bayesian side is not a slide — it's the machinery I use. My **LLM Ensemble
Bias Detection** project runs a PyMC hierarchical model with partial pooling,
MCMC diagnostics (R-hat < 1.01, ESS), 95% HDI credible intervals, and bootstrap CIs
across 67,500 ratings, cross-validated against Friedman χ² = 42.73 (p < 0.001).
The third project — an LLM-ensemble red-team framework at Krippendorff's α = 0.81
and 340× cost reduction — shows the same rigor applied to a very different domain,
which is what a real-time clinical decision-support product tends to need.

Happy to walk through calibration, uncertainty quantification, and the deployment
stack in a first call.

Derek Lankeaux · [LinkedIn](https://linkedin.com/in/derek-lankeaux) ·
[GitHub](https://github.com/dl1413)
