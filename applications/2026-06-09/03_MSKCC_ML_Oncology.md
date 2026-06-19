# Memorial Sloan Kettering — Machine Learning Engineer, Computational Oncology

**Location:** New York, NY (on-site/hybrid) · **Lead project:** Project 3 (Breast Cancer Classification)
**Source:** careers.mskcc.org · **Snippet base:** APPLICATION_SNIPPETS §3 (Healthcare ML)

## JD keyword echo (≥3 verbatim)
- "clinical decision support", "model calibration", "explainability", "SHAP", "fairness audit", "regulated"

## Resume Projects-section order
1. **Breast Cancer ML Classification** (LEAD)
2. AI Safety Red-Team Evaluation (rigor / governance signal)
3. LLM Ensemble Bias Detection (statistical rigor)

## Tailored resume bullets
- Shipped clinical-grade ensemble breast-cancer classifier exceeding the 90-95% human-expert range: **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987**
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting + 2) under nested cross-validation; selected AdaBoost on held-out evidence
- Platt-calibrated probabilities reducing **ECE by 71.5% (0.0312 -> 0.0089)**; context-adaptive threshold tuning (100% sensitivity at 0.31 for mass screening)
- Bayesian hyperparameter search (Optuna TPE) converged in 45 trials vs 240 grid-search trials (5x reduction)
- VIF multicollinearity pruning, SMOTE class balancing, RFE feature selection; SHAP attributions per prediction for clinical transparency (IEEE 2830-2025)
- Deployed behind FastAPI + MLflow registry; **<100ms p95 latency**, audit-trailed for regulated review

## Cover letter

> Dear MSKCC Computational Oncology team,
>
> The work most relevant to this role is a clinical-grade breast-cancer
> classifier I shipped this year. I benchmarked 8 algorithms end-to-end —
> Random Forest through stacking ensembles — under nested cross-validation
> and landed AdaBoost at **99.12% accuracy, 100% precision (zero false
> positives), 98.59% recall, and ROC-AUC 0.9987**, comfortably above the
> 90-95% range typically cited for expert reads. Just as important for
> deployment in a regulated clinical setting: Platt calibration reduced
> ECE by 71.5%, threshold tuning hits 100% sensitivity at the screening
> operating point, and every prediction ships with a SHAP explanation and
> MLflow-registered model card aligned with **IEEE 2830-2025** transparency
> requirements. The service sits behind FastAPI at <100ms p95.
>
> The methodology generalizes — I've applied the same statistical discipline
> to an LLM red-team eval (96.8% accuracy, Krippendorff's alpha = 0.81)
> and a multi-LLM bias study (Friedman chi-squared = 42.73, p < 0.001 via
> PyMC hierarchical model with partial pooling) — but the breast-cancer
> pipeline is the one that maps directly to computational oncology: tabular
> clinical signal, high-stakes asymmetric error costs, and a deployment
> bar that demands calibration, explainability, and a fairness audit, not
> just AUC.
>
> Finishing my MS in Applied Statistics at RIT (2026). NYC-based / on-site
> available. Authorized to work in the US.
>
> Best,
> Derek Lankeaux
