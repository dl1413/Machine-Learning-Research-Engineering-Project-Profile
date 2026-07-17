# Apex Systems — Data Scientist (Fully Remote) — Healthcare Analytics

**Location:** Remote (US) • **Fit:** ⭐⭐ Good
**Apply:** https://www.apexsystems.com/job/3028454_usa/data-scientist-fully-remote

**Why this fits:** Apex is placing a fully remote DS to build predictive models and analytics on large healthcare datasets, using stats/ML to inform clinical, operational, and financial decisions. My Breast Cancer Classification project is directly on-thesis: clinical predictive modeling, calibrated probabilities, threshold policies for context-specific decisions (screening vs. diagnostic), and IEEE 2830-2025 fairness auditing. Master's + strong Python/SQL + XGBoost/tree-based models is exactly the stated requirement.

**Lead project:** Breast Cancer Classification
**Supporting:** LLM Bias Detection (Bayesian rigor + reproducibility standards), AI Safety Red-Team (production MLOps at scale)

---

## Cover Letter

Dear Apex Systems Team,

I'm applying for the fully remote Data Scientist role because my recent work has been exactly the shape of what the posting describes: predictive modeling on healthcare data, translated into decisions clinical and operational stakeholders can actually use.

My clinical-grade breast cancer classification project reached 99.12% accuracy with 100% precision (zero false positives) and 98.59% recall, ROC-AUC 0.9987. But the numbers that matter for real clinical use are the ones downstream of accuracy: I calibrated probabilities with Platt scaling and cut expected calibration error from 0.0312 to 0.0089 — a 71.5% reduction — so predicted probabilities can actually drive triage policy. I then tuned decision thresholds separately for mass screening (100% sensitivity at threshold 0.31) versus confirmatory diagnostics (higher precision). That's the workflow a healthcare team needs: not a single threshold, but a calibrated model with a decision policy tuned per use case.

Upstream, I built a rigorous preprocessing pipeline: VIF multicollinearity analysis, SMOTE class balancing, RFE feature selection, and a comparative benchmark across eight algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, plus Stacking and Voting ensembles). I used Optuna's Tree-structured Parzen Estimator to converge on hyperparameters in 45 trials versus a 240-trial grid — a 5× reduction that keeps model iteration inexpensive. And I shipped SHAP-based explanations plus IEEE 2830-2025-aligned model cards so clinical reviewers can audit any prediction.

Production: MLflow registry, FastAPI serving at <100ms p95 latency. I'm finishing my MS in Applied Statistics at RIT (2026) and would love to bring this stack to Apex.

Best,
Derek Lankeaux

---

## Resume-Bullet Variant

- Delivered clinical-grade ensemble classifier reaching 99.12% accuracy, 100% precision, 98.59% recall (ROC-AUC 0.9987) — exceeding human expert performance (90–95%) on the same task
- Cut expected calibration error 71.5% (0.0312 → 0.0089) via Platt scaling; tuned decision thresholds per clinical policy (100% sensitivity at 0.31 for screening, higher precision for diagnostic)
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting); Optuna TPE converged in 45 trials vs 240 for grid (5× efficiency); preprocessing pipeline included VIF, SMOTE, RFE
- Shipped production stack: MLflow model registry, FastAPI serving at <100ms p95 latency, SHAP-based explanations, IEEE 2830-2025 fairness-audited model cards

---

## 60-Second Hook

"I built a clinical-grade classifier that hit 99.12% accuracy and 100% precision — but the piece that matters for real healthcare use is downstream: I calibrated probabilities so ECE dropped 71%, and I tuned thresholds separately for screening versus diagnostic contexts, because one number doesn't work for both. Full production stack — MLflow registry, FastAPI at sub-100ms latency, SHAP explanations, IEEE 2830-2025 model cards. That's the workflow I'd bring to your healthcare analytics team."
