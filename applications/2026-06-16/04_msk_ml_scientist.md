# Application 4 — Memorial Sloan Kettering, Machine Learning Scientist

- **Date:** 2026-06-16
- **Location:** New York, NY (on-site)
- **Source:** careers.mskcc.org
- **Lead project:** Breast Cancer Classification
- **Supporting:** AI Safety Red-Team (rigor, audit), LLM Bias Detection (Bayesian)

## Resume bullet order

1. **Clinical-Grade Breast Cancer Classification** — 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting); winning model at 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987; SHAP per-prediction explanations; FastAPI <100ms p95; IEEE 2830-2025 transparency.
2. **AI Safety Red-Team Evaluation** — Krippendorff alpha = 0.81 across 3-LLM ensemble; demonstrates the same audit-grade rigor on a different domain.
3. **LLM Ensemble Bias Detection** — PyMC Bayesian hierarchical model with R-hat < 1.01; 95% HDI quantification.

## Cover letter

Dear MSK Machine Learning team,

The work most relevant to MSK is a clinical-grade breast-cancer classifier I shipped this year. I benchmarked 8 algorithms end-to-end — Random Forest through stacking ensembles — and landed at 99.12% accuracy with 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987, comfortably above the 90-95% range typically cited for human expert reads. Just as important for clinical deployment: the pipeline ships with SHAP explanations per prediction, VIF-pruned features, SMOTE-balanced training, Platt-calibrated probabilities (ECE reduced 71.5% to 0.0089), and a FastAPI service under 100ms p95 — all aligned with IEEE 2830-2025 transparency standards and adaptable to the audit requirements clinical AI faces at MSK.

Two habits I'd bring beyond the headline metrics: (1) context-adaptive thresholds — I tuned the decision threshold to 0.31 for a 100% sensitivity screening mode, the kind of policy decision that has to be explicit in clinical settings; (2) Bayesian uncertainty quantification — my LLM bias-detection work used a PyMC partial-pooling hierarchical model (R-hat < 1.01, ESS > 1000) to produce 95% HDI credible intervals rather than point estimates. Computational oncology cares about "how sure are we" as much as "what's the prediction," and I default to answering both.

I'm NYC-available, finishing my MS in Applied Statistics at RIT (2026), authorized to work in the US.

Best,
Derek Lankeaux

## JD keywords to echo

`clinical decision support`, `oncology`, `EHR`, `medical imaging`, `model interpretability`, `SHAP`, `calibration`, `regulatory`, `validation`
