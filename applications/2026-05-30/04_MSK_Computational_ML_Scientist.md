# Memorial Sloan Kettering — Computational ML Scientist (Computational Oncology)

**Location:** New York, NY
**Role family:** Healthcare / Clinical ML
**Lead project:** Breast Cancer Classification (Clinical-grade ensemble)
**Supporting:** AI Safety Red-Team (rigor, audit trails), LLM Bias Detection (statistical defensibility)
**JD link:** https://careers.mskcc.org/ (paste exact posting URL on submit)
**Resume version:** Resume_Derek_Lankeaux_v_healthcare.pdf
**Status:** ready_to_submit

---

## Cover Letter

Dear MSK Hiring Team,

I'm applying for the Computational ML Scientist role because the work I'm most proud of is also the work that most needs the kind of clinical and statistical scrutiny MSK brings.

I built a clinical-grade breast-cancer classifier by benchmarking 8 algorithms end-to-end — Random Forest, XGBoost, LightGBM, AdaBoost, plus stacking and voting ensembles — under nested cross-validation with Optuna Bayesian hyperparameter search (45 trials vs. 240 for grid). The winning AdaBoost ensemble landed at **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** on the held-out test set, comfortably above the 90–95% range typically cited for human expert reads. The numbers only matter because of the discipline around them: VIF-based multicollinearity pruning, SMOTE class-balancing, RFE feature selection, Platt-calibrated probabilities (**ECE 0.0089, a 71.5% reduction**), and threshold tuning for **100% sensitivity at the screening operating point** — because in screening, recall is the deliverable.

For deployment, the model ships behind a FastAPI service at **<100ms p95 latency**, with MLflow model registry, per-prediction SHAP explanations to satisfy IEEE 2830-2025 transparency requirements, and a fairness audit. Cross-validation stability holds at 98.46% ± 1.12%, so the result is reproducible, not lucky.

The supporting work — multi-LLM evaluation pipelines with Krippendorff's alpha at 0.81–0.84 and Bayesian hierarchical inference — shows the same statistical defensibility carries into messier, less-clean domains, which is most of clinical AI.

I'd love to bring this combination of rigorous model selection, calibration-first deployment, and explainability discipline to MSK's computational oncology work. Available on-site in NYC; targeting a 2026 start after my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux

---

## Top Resume Bullets

- Benchmarked 8 algorithms under nested cross-validation; shipped AdaBoost ensemble at 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987 — exceeding 90-95% human-expert range
- Platt-calibrated probability outputs (ECE 0.0089, 71.5% reduction); threshold-tuned to 100% sensitivity at 0.31 for mass-screening operating point
- Deployed containerized FastAPI service with MLflow registry; <100ms p95 latency, per-prediction SHAP explanations, IEEE 2830-2025-aligned audit trail
- Built clinical-tabular preprocessing pipeline: VIF multicollinearity pruning, SMOTE class-balancing, RFE feature selection, stratified cross-validation

## JD Keywords to Echo
Clinical decision support, healthcare ML, ensemble methods, model calibration, SHAP, explainability, IEEE 2830-2025, fairness audit, FastAPI deployment, MLflow, reproducible research.
