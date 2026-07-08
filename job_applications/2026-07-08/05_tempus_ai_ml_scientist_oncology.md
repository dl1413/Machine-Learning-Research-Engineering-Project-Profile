# Tempus AI — Machine Learning Scientist, Oncology

**Posting:** https://www.tempus.com/careers/ (search: ML Scientist / Oncology)
**Location:** Remote (US)
**Date drafted:** 2026-07-08

---

Dear Tempus AI Oncology ML team,

I'm applying for the Machine Learning Scientist (Oncology) role. Tempus's core bet — that better molecular and clinical data can turn oncology decisions into calibrated, defensible probabilities rather than gestalt calls — is exactly the problem I've been working on. My most recent clinical ML project is a direct fit.

**Directly relevant work:**

- **Clinical-Grade Breast Cancer ML Classification (2026).** I built an ensemble system that reached **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** on the WDBC diagnostic dataset — exceeding published human-expert performance (90–95%). It's the full clinical pipeline, not a leaderboard number: VIF multicollinearity screen, SMOTE class balancing, RFE feature selection, and an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting). Bayesian hyperparameter search with Optuna TPE converged in 45 trials vs 240 for grid search.
- **Calibration and decision policies.** Platt scaling reduced Expected Calibration Error 71.5% (0.0312 → 0.0089) so downstream clinicians can trust the probabilities, not just the top-1 label. Context-adaptive thresholds — e.g., 100% sensitivity at 0.31 for mass screening vs a specificity-tuned threshold for confirmatory read — the same policy machinery Tempus applies for screening vs treatment-planning use.
- **Explainability + governance.** SHAP values for per-prediction transparency, fairness auditing aligned with IEEE 2830-2025, EU AI Act-compatible model cards. Deployed via MLflow registry + FastAPI (<100ms p95). That's the artifact set regulators and clinical partners actually ask for.

**Adjacent strengths.** Applied Statistics MS with Bayesian hierarchical modeling (PyMC, MCMC R-hat < 1.01, 95% HDI) and LLM-ensemble evaluation (Krippendorff's α = 0.81 across GPT-4o / Claude-3.5 / Llama-3.2) — useful whether the team is scaling structured predictions from EHR/genomic signal or extracting features from unstructured pathology or radiology text.

Portfolio and code: https://dl1413.github.io/LLM-Portfolio/ • https://github.com/dl1413. Resume attached.

Thank you,
Derek Lankeaux
