# 05 — Memorial Sloan Kettering, ML Engineer, Computational Oncology

- **Location:** New York City (on-site / hybrid)
- **Source:** careers.mskcc.org (direct)
- **Role family:** Healthcare / Clinical ML
- **Lead project:** Clinical-Grade Breast Cancer ML Classification
- **Supporting project:** AI Safety Red-Team Evaluation (rigor / compliance)
- **Resume version:** `resume_v3_healthcare.pdf`

## Cover letter (paste-ready)

Hi MSK team,

The work most relevant to MSK's Computational Oncology group is a
clinical-grade breast-cancer classifier I shipped this year. I benchmarked
8 algorithms end-to-end — Random Forest, XGBoost, LightGBM, AdaBoost,
Stacking, Voting, and two others — and landed at 99.12% accuracy with
100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987,
comfortably above the 90-95% range typically cited for human expert reads.
Just as important for clinical deployment: the pipeline ships with SHAP
explanations per prediction, VIF-pruned features, SMOTE class balancing,
and a FastAPI service under 100ms p95, all aligned with IEEE 2830-2025
transparency standards and the EU AI Act's high-risk system requirements.

The same statistical discipline shows up in my LLM red-team work — a
3-model ensemble with 47 engineered features at 96.8% accuracy and
Krippendorff's alpha = 0.81 — but the breast-cancer project is the one
that maps most directly to your work. I'd love to apply the same rigor to
MSK's clinical-AI problems and would be excited about an NYC role
starting 2026 once I wrap my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux
LinkedIn: linkedin.com/in/derek-lankeaux | GitHub: github.com/dl1413

## Resume top-3 bullets

1. Built clinical-grade classifier at 99.12% accuracy / 100% precision (zero false positives) / 98.59% recall / ROC-AUC 0.9987, exceeding the 90-95% human-expert range
2. Implemented SHAP-based per-prediction explanations satisfying IEEE 2830-2025 transparency requirements for clinical decision support
3. Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation, RFE feature selection

## JD keyword targets

- [ ] "clinical" / "diagnostic"
- [ ] "EHR" / "imaging" (whichever the JD specifies)
- [ ] "explainability" / "SHAP"
- [ ] "HIPAA" / "regulatory"
- [ ] "production" / "deployment"
