# 04 — Memorial Sloan Kettering · Computational Oncology ML Engineer

- **Location:** NYC (on-site)
- **Family:** Healthcare / Clinical ML
- **Lead project:** P3 — Clinical-Grade Breast Cancer ML Classification
- **Supporting:** P1 Red-Team (eval rigor), P2 Bias (Bayesian stats)
- **JD link:** https://careers.mskcc.org/ (Computational Oncology ML Engineer)
- **Resume version:** `resume_v3_healthcare.pdf`
- **Cover letter:** YES

## JD keywords to echo verbatim (≥3)
1. "clinical"
2. "explainability" / "SHAP"
3. "FDA / regulatory / IEEE" (transparency)
4. "feature engineering"
5. "cross-validation"

## Resume bullet stack

**Breast Cancer Classification (lead — Healthcare ML variant)**
- Built clinical-grade classifier exceeding the 90–95% human-expert range — 99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation
- Productionized winning ensemble behind containerized FastAPI service with MLflow model registry; p95 latency under 100ms

**AI Safety Red-Team Evaluation (supporting — eval rigor)**
- 96.8% accuracy ensemble auto-grading 12,500 response pairs with Krippendorff α = 0.81 and full IEEE 2830-2025 audit trail

**LLM Bias Detection (supporting — statistical defensibility)**
- PyMC partial-pooling hierarchy with R-hat < 1.01, ESS > 1000; Friedman χ² = 42.73, p < 0.001

## Cover letter

> Dear MSK Computational Oncology hiring team,
>
> The work most relevant to your team is a clinical-grade breast-cancer
> classifier I shipped this year. I benchmarked 8 algorithms end-to-end —
> Random Forest through stacking ensembles — and landed at 99.12% accuracy
> with 100% precision (zero false positives), 98.59% recall, and ROC-AUC
> 0.9987, comfortably above the 90–95% range typically cited for human
> expert reads. Just as important for clinical deployment: the pipeline
> ships with SHAP explanations per prediction, VIF-pruned features, SMOTE
> class-balancing, stratified cross-validation, and a FastAPI service under
> 100ms p95, all aligned with IEEE 2830-2025 transparency standards.
>
> Two other recent projects show the same standard applied beyond
> oncology. An AI Safety Red-Team evaluation framework auto-grades 12,500
> response pairs at 96.8% accuracy and ROC-AUC 0.9923 under a full audit
> trail. And a textbook-bias study runs a PyMC Bayesian hierarchical model
> (partial pooling, R-hat < 1.01, ESS > 1000) with Friedman χ² = 42.73,
> p < 0.001 — the kind of statistical defensibility I'd want to bring to
> any clinical inference where a result has to survive review.
>
> I'm based in / available for New York City, targeting a 2026 start once
> I wrap my MS in Applied Statistics at RIT. I'd love to apply this rigor
> to MSK's computational oncology problems — pathology, biomarker
> discovery, or clinical-decision-support — where 100% precision actually
> matters.
>
> Best,
> Derek Lankeaux
