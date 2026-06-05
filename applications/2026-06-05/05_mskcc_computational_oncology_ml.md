# Memorial Sloan Kettering — Computational Oncology, ML Engineer

**Date:** 2026-06-05
**Location:** NYC (on-site)
**Source:** mskcc.org/careers
**Lead project:** Project 3 — Clinical-Grade Breast Cancer Classification
**Role family:** Healthcare / Clinical ML
**Resume version:** resume_v3_clinical.pdf

---

## JD keywords to mirror
- "clinical" / "diagnostic"
- "explainability" / "SHAP"
- "ensemble"
- "FastAPI" / "production"
- "transparency" / "audit"

## Cover letter opener (metric hook)

> The work most relevant to MSK's Computational Oncology team is a
> clinical-grade breast-cancer classifier I shipped this year. I
> benchmarked **8 algorithms** end-to-end — Random Forest through stacking
> ensembles — and landed at **99.12% accuracy with 100% precision (zero
> false positives), 98.59% recall, and ROC-AUC 0.9987**, comfortably above
> the 90-95% range typically cited for human expert reads. Just as
> important for clinical deployment: the pipeline ships with SHAP
> explanations per prediction, VIF-pruned features, and a FastAPI service
> under **100ms p95**, all aligned with IEEE 2830-2025 transparency
> standards. I'd love to apply the same rigor to MSK's diagnostic-AI
> pipelines.

## Resume bullets to surface

- Built clinical-grade classifier exceeding the 90-95% human-expert range, with zero false positives across the held-out test set
- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) and shipped winning ensemble at 99.12% accuracy / ROC-AUC 0.9987
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency for clinical decision support
- Deployed as containerized FastAPI service with MLflow model registry; p95 latency under 100ms

## Supporting projects
- AI Safety Red-Team Eval — same rigor on LLM eval (Krippendorff α = 0.81, IEEE 2830-2025 audit trail)
- LLM Bias Detection — Bayesian PyMC hierarchical inference, p < 0.001

## Submission checklist
- [ ] Resume reordered: Breast Cancer project on top, MSK-style framing
- [ ] Cover letter opens with 99.12% / 100% precision / SHAP hook
- [ ] "clinical", "SHAP", "explainability" verbatim
- [ ] Highlight IEEE 2830-2025 + ISO/IEC 23894:2025 compliance
- [ ] Salary expectation: open, targeting NYC market
