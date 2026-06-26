# 03 — Memorial Sloan Kettering | Computational Oncology ML Engineer

**Location:** New York City (on-site / hybrid)
**Source:** careers.mskcc.org
**Lead project:** P3 — Clinical-Grade Breast Cancer Classifier
**Supporting:** P1 Red-Team (eval rigor, IEEE 2830-2025), P2 (Bayesian stats)
**Role family:** Healthcare / Clinical ML

---

## JD keywords to mirror

- "clinical decision support"
- "diagnostic"
- "explainability" / "SHAP"
- "model validation"
- "tabular clinical data"
- "fairness"
- "IRB / audit"

## Resume reordering

1. **Breast Cancer Classification** *(top)*
2. AI Safety Red-Team Evaluation Framework
3. LLM Ensemble Bias Detection

## Top 4 resume bullets (paste under Project 3)

- Built clinical-grade ensemble classifier reaching **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** — above the 90–95% range typically cited for human-expert reads
- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting +2) under nested cross-validation; selected winner on calibrated probability and recall, not accuracy alone
- Implemented per-prediction **SHAP** explanations and a **fairness** audit to satisfy IEEE 2830-2025 transparency requirements for **clinical decision support**
- Productionized as containerized FastAPI service with MLflow model registry, p95 latency under 100ms — preprocessing covers VIF multicollinearity diagnostics, SMOTE class balancing, stratified CV

## Supporting bullets

- (Red-Team Eval) Same audit discipline applied to LLM safety: Krippendorff alpha 0.81, full IEEE 2830-2025 audit trail across 12,500 evals
- (Bias Detection) PyMC Bayesian hierarchical model with R-hat < 1.01, 95% HDI credible intervals — same statistical machinery I'd bring to **model validation** on clinical cohorts

## Cover letter (paste-ready)

> Dear MSK Computational Oncology Team,
>
> The work most relevant to this role is a clinical-grade breast-cancer
> classifier I shipped this year. I benchmarked 8 algorithms end-to-end —
> Random Forest through stacking ensembles — under nested cross-validation
> and landed at **99.12% accuracy with 100% precision (zero false
> positives), 98.59% recall, and ROC-AUC 0.9987**, comfortably above the
> 90–95% range typically cited for human-expert reads. Just as important
> for **clinical decision support** deployment: the pipeline ships with
> per-prediction **SHAP** explanations, VIF-pruned features, SMOTE class
> balancing, and a FastAPI service under 100ms p95, all aligned with
> IEEE 2830-2025 transparency standards.
>
> Two other projects round out the picture for an oncology setting.
> First, a PyMC Bayesian hierarchical bias study (R-hat < 1.01,
> 95% HDI credible intervals, Friedman chi-squared = 42.73, p < 0.001) —
> the same machinery I'd apply to **model validation** across MSK patient
> subgroups. Second, an AI Safety Red-Team Evaluation framework
> (Krippendorff alpha = 0.81, full audit trails across 12,500 evals)
> that taught me how to keep an evaluation defensible when stakes are
> high. Both habits — uncertainty quantification and audit-grade
> documentation — feel especially relevant to clinical ML at MSK.
>
> I'm based in NYC and would be on-site any cadence MSK prefers,
> targeting a 2026 start once I wrap my MS in Applied Statistics at RIT.
> Portfolio, code, and the technical reports are on GitHub (dl1413).
>
> Best,
> Derek Lankeaux

## Checklist

- [ ] P3 at top of Projects section
- [ ] Phrases echoed: "clinical decision support", "SHAP", "model validation", "fairness", "tabular clinical data"
- [ ] Hook: 99.12% / 100% precision / <100ms p95
- [ ] LinkedIn includes "Healthcare AI" tag
- [ ] Mention IRB / HIPAA awareness in skills section if JD calls for it
- [ ] Salary: "open"
- [ ] JD PDF saved
