# Bayesian Health — Senior ML Data Scientist

**Location:** Remote (US)
**Posting:** https://www.remoterocketship.com/us/company/bayesianhealth/
**Lead project:** Breast Cancer ML Classification + LLM Bias Detection Bayesian hierarchical pipeline

---

## Cover Letter

Hi Bayesian Health team,

The Senior ML Data Scientist role at a company built on adaptive Bayesian modeling for real-world clinical care is the closest match I've seen to how I actually work.

**Clinical-grade ensemble modeling.** My **Breast Cancer ML Classification** system (April 2026) benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) and shipped an AdaBoost model at **99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987** — meaningfully above the 90–95% band of human expert performance. I used Optuna TPE hyperparameter search (converging in 5× fewer trials than grid search: 45 vs 240), VIF-based multicollinearity screening, SMOTE for class balance, RFE feature selection, and Platt calibration (ECE 0.0312 → 0.0089, a 71.5% reduction) so the model's confidence is trustworthy at deployment. Threshold optimization lets the same model serve two decision policies: **100% sensitivity at 0.31 for population screening**, high-precision at 0.67 for confirmatory diagnosis. FastAPI deployment under 100 ms p95, MLflow model registry, SHAP for clinical transparency, IEEE 2830-2025 compliance.

**Bayesian hierarchical modeling in the wild.** My **LLM Ensemble Bias Detection** work used PyMC + partial pooling on **67,500 ratings** with **MCMC R-hat < 1.01, 95% HDI, ArviZ diagnostics**, and bootstrap CIs to flag **12.3% high-uncertainty passages** for expert review. The same pattern — partial pooling across patients, sites, or care pathways — is exactly what Bayesian Health's adaptive care augmentation calls for.

**Production hygiene.** Circuit breakers, exponential backoff, MLflow experiment tracking; 80K+ API calls / 2.5M tokens processed in the LLM pipeline without a lost run.

Applied Statistics MS at RIT (expected 2026), fully remote-friendly, US-authorized.

Would love to talk about applying partial-pooling + calibration + decision-policy work to your platform.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Attach

- Resume PDF
- `Breast_Cancer_Classification_Publication.pdf`
- `LLM_Bias_Detection_Publication.pdf`
