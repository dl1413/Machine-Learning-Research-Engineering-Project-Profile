# Breast Cancer ML Classification — Bullet Variants

Project header line:
> **Clinical-Grade Breast Cancer ML Classification System** — Independent Research · January 2026
> Tech: scikit-learn, XGBoost, LightGBM, AdaBoost, Optuna, SMOTE, SHAP, MLflow, FastAPI

---

## Research / Frontier-lab framing

**Short**
- Developed ensemble ML system achieving **99.12% accuracy**, **100% precision**, ROC-AUC 0.9987 — exceeding published human-expert performance (90–95%) on the benchmark.
- Optuna TPE hyperparameter optimization converged in **45 trials vs 240 for grid search** (5× faster); Platt calibration reduced ECE 71.5% (0.0312 → 0.0089).

**Medium**
- Benchmarked **8 algorithms** head-to-head (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, Gradient Boosting, Logistic Regression) — AdaBoost won at 99.12% accuracy.
- Applied advanced preprocessing pipeline: VIF multicollinearity analysis, SMOTE class balancing, RFE feature selection.
- Bayesian hyperparameter optimization (Optuna TPE) converged in 45 trials vs. 240 for grid search.
- Calibrated probabilities with Platt scaling (ECE 0.0312 → 0.0089, 71.5% improvement) and SHAP explainability for clinical fairness audit per IEEE 2830-2025.

---

## AI Safety / Alignment framing

**Short**
- Demonstrated rigorous statistical evaluation methodology on a high-stakes classification benchmark: 99.12% accuracy with **100% precision** (zero false positives), Platt-calibrated probabilities (ECE 0.0089), SHAP explainability, and IEEE 2830-2025 fairness auditing.

> Note: This project is usually the weakest fit for safety/alignment roles. Use the AI Safety Red-Team and Bias Detection bullets as your primaries and keep this one in reserve as a tertiary "I do statistical rigor at high-stakes" credential.

---

## Applied / Production ML framing

**Short**
- Deployed clinical-grade classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987) via FastAPI at **<100ms p95 latency** with MLflow registry and Platt-calibrated probabilities.

**Medium**
- Shipped clinical-grade ensemble classifier via FastAPI at **<100ms p95 latency** — 99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987.
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting, GBM, Logistic) before selecting AdaBoost on the accuracy/latency frontier.
- **Optuna TPE hyperparameter search** converged in 45 trials vs. 240 for grid (5× faster) with reproducible random seeds.
- **Platt calibration** reduced expected calibration error 71.5% (0.0312 → 0.0089) for clinically actionable probability outputs; context-adaptive thresholds (100% sensitivity at 0.31 for screening).
- Full MLflow model registry with versioning, model cards, and rollback support.

**Long** — adds:
- VIF multicollinearity analysis, SMOTE class balancing, and RFE feature selection in preprocessing.
- SHAP explanations exposed as endpoints for clinical transparency and fairness auditing.
- IEEE 2830-2025 (Transparent ML) compliance documentation.

---

## ML Platform / Infra framing

**Short**
- Deployed FastAPI inference endpoint at **<100ms p95 latency** with MLflow model registry, Platt-calibrated probabilities (ECE 0.0089), and SHAP explanation endpoints.

**Medium**
- Built FastAPI inference service achieving <100ms p95 latency for ensemble classifier (AdaBoost over 30 features).
- Integrated MLflow model registry for versioning, model cards, staging/production transitions, and rollback.
- Exposed SHAP explanation endpoints alongside prediction endpoints — explanations served per-request for stakeholder review.
- Documented Optuna TPE hyperparameter search trajectory (45 trials) and reproducibility metadata (random seeds, environment pinning).

---

## Data Scientist / Quant / Stats framing

**Short**
- Validated ensemble on 8-algorithm benchmark with stratified k-fold CV (98.46% ± 1.12%), Optuna TPE hyperparameter search (45 trials, 5× faster than grid), and Platt calibration (ECE 0.0089).
- 99.12% accuracy, 100% precision, ROC-AUC 0.9987 — exceeding published human-expert range (90–95%).

**Medium**
- Built 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting, GBM, LR) with stratified k-fold cross-validation — CV stability 98.46% ± 1.12%.
- Used **Optuna TPE Bayesian hyperparameter search** (45 trials) instead of grid (240 trials) — converged 5× faster with reproducible seeds.
- Applied preprocessing diagnostics: **VIF multicollinearity analysis** (drop features with VIF > 10), **SMOTE** for class balancing on minority class, **RFE** for feature selection (30 features retained).
- Calibrated probabilities with **Platt scaling**, reducing ECE 0.0312 → 0.0089 (71.5% reduction) for clinically actionable confidence outputs.
- Optimized operating thresholds for context: 100% sensitivity achieved at threshold 0.31 for mass screening use case.
- Reported with explicit comparison to human-expert range (90–95%) and reproducibility documentation per IEEE 2830-2025.

---

## Civic / Government / Cross-source data framing

> Use this framing for civic-data and policy-analytics roles like the NYC OCCECE Data Manager.

**Short**
- Built reproducible analytical pipeline on a public health benchmark dataset: preprocessing audit (VIF, SMOTE, RFE), 8-algorithm comparison, calibrated outputs (ECE 0.0089), and SHAP explanations for non-technical reviewers.

**Medium**
- Built an end-to-end reproducible analytical pipeline for a public-health classification benchmark — from raw data through preprocessing diagnostics (VIF, SMOTE, RFE) to deployed model with calibration and explanation layers.
- Documented preprocessing decisions with clear rationale (multicollinearity thresholds, class balancing approach, feature selection criteria) so a reader can replicate or audit each step.
- Reported model outputs with **calibrated probabilities** (Platt scaling, ECE 0.0089) so that probabilistic statements to non-technical stakeholders are actually meaningful, not arbitrary.
- Used **SHAP explanations** to communicate per-prediction reasoning to non-technical reviewers (clinical stakeholders, fairness auditors) — same explainability layer applicable to civic decision-support contexts.
- Reproducibility metadata (random seeds, environment pinning, MLflow tracking) supports future re-runs and audits.
