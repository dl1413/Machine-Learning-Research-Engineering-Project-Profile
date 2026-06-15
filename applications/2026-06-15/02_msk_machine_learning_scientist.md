# 02 — Memorial Sloan Kettering · Machine Learning Scientist

- **Date pulled:** 2026-06-15
- **Location:** New York, NY (on-site, MSK main campus)
- **Source:** careers.mskcc.org (req 2025-86362)
- **JD link:** https://careers.mskcc.org/vacancies/2025-86362-machine-learning-scientist/
- **Lead project:** Project 3 — Clinical-Grade Breast Cancer ML Classification
- **Supporting:** Project 1 (rigor / audit trails), Project 2 (Bayesian uncertainty)
- **Resume version to send:** `resume_v3_healthcare.pdf`
- **Cover letter:** Yes (healthcare lab — yes, attach)

## JD signals (from public summary)

> "Combine advanced statistical methods, deep learning, and high-performance
> computing to extract insights from complex datasets… translate research
> into production-ready AI tools used in healthcare environments. Partner
> closely with clinicians… proficiency in PyTorch and distributed training,
> proven experience with model deployment in high-stakes environments."

Keywords to echo: `clinical`, `high-stakes`, `production-ready`, `model
deployment`, `statistical methods`, `clinicians`, `medical imaging`,
`computational pathology`.

## Tailored cover-letter opener (paste-ready)

> The work most relevant to MSK's Computational Oncology group is a
> clinical-grade breast-cancer classifier I shipped this year. I
> benchmarked 8 algorithms end-to-end — Random Forest through stacking
> ensembles — and landed at **99.12% accuracy with 100% precision (zero
> false positives), 98.59% recall, and ROC-AUC 0.9987**, comfortably above
> the 90–95% range typically cited for human expert reads. Just as
> important for clinical deployment: the pipeline ships with SHAP
> explanations per prediction, VIF-pruned features, Platt-calibrated
> probabilities (ECE 0.0089), and a FastAPI service under 100ms p95, all
> aligned with IEEE 2830-2025 transparency standards. I'd love to apply
> the same rigor to MSK's high-stakes diagnostic problems.

## Resume bullets to surface

(from `APPLICATION_SNIPPETS.md` → Project 3 → Healthcare ML)

- Built clinical-grade classifier exceeding the 90–95% human-expert range, with zero false positives across the held-out test set
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation
- Productionized winning model behind FastAPI with MLflow model registry and <100ms p95 latency

## Note on PyTorch / distributed training (JD gap to address)

JD asks for PyTorch + distributed training; portfolio is heavier on
scikit-learn/XGBoost. In cover letter, explicitly mention: *"Currently
extending the pipeline to PyTorch + DDP for image-based pathology inputs
as part of MS coursework — happy to discuss the migration plan."*

## Application checklist

- [x] Lead project surfaced first
- [x] 3+ JD phrases echoed (`clinical`, `high-stakes`, `production-ready`)
- [x] Metric hook in opener (99.12%, 100% precision, ROC-AUC 0.9987)
- [x] Gap addressed honestly (PyTorch/DDP)
- [x] Work-auth: US authorized
- [x] Salary expectation: open, targeting NYC academic-medical-center market
