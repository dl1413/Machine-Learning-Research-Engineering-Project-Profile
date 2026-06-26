# Memorial Sloan Kettering — ML Engineer, Computational Oncology (NYC)

- **Role family:** Healthcare / Clinical ML
- **Lead project:** P3 — Clinical-Grade Breast Cancer ML Classification
- **Supporting:** P1 — AI Safety Red-Team (rigor/eval) · P2 — LLM Bias Detection (stats)
- **Source:** mskcc.org/careers (Tier D)
- **Resume version:** `resume_v3_clinical` (Projects section reordered → P3 first)

## JD KEYWORDS TO ECHO (fill from live JD before submit)
`__________`, `__________`, `__________`
Bank: clinical decision support, diagnostic ML, SHAP explainability,
fairness auditing, computational pathology, model validation.

## Resume Projects-section order
1. **Breast Cancer Classification** — *Healthcare / Clinical ML* bullets
2. AI Safety Red-Team Evaluation — *ML Research Engineer* bullets (eval rigor)
3. LLM Ensemble Bias Detection — *Data Scientist (Bayesian)* bullets (stats depth)

## Cover letter

Dear MSK Hiring Team,

Trained an 8-algorithm benchmark ensemble that hits 99.12% accuracy, 100%
precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987 on
breast-cancer classification — above the 90–95% human-expert range — and
deployed it as a <100ms p95 FastAPI service.

The work most relevant to MSK's computational-oncology mission is a
clinical-grade breast-cancer classifier I shipped this year. I benchmarked 8
algorithms end-to-end — Random Forest through stacking ensembles — and landed
at the numbers above. Just as important for clinical deployment: the pipeline
ships with SHAP explanations per prediction, VIF-pruned features, SMOTE
class-balancing, and a FastAPI service under 100ms p95, all aligned with IEEE
2830-2025 transparency standards and a fairness audit.

My other two projects show that this rigor is a habit, not a one-off. An
independent AI Safety red-team eval framework (96.8% accuracy, ROC-AUC 0.9923
over a 12,500-pair benchmark) demonstrates disciplined model validation and
uncertainty quantification, and an LLM bias-detection study used a PyMC
hierarchical model (R-hat < 1.01, 95% HDI) to defend a finding at p < 0.001 —
the kind of statistical care clinical claims require.

I'm based in / available for New York City and open to hybrid, targeting a 2026
start once I wrap my MS in Applied Statistics at RIT. Portfolio, code, and the
three technical reports are on my GitHub (dl1413). I'd love to apply the same
rigor to MSK's diagnostic-AI problems.

Best,
Derek Lankeaux

## Pre-submit checklist
- [ ] 3+ JD phrases echoed in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio links live
- [ ] Work auth: Authorized to work in the US
- [ ] Salary: open, targeting market for role/location
- [ ] JD PDF saved in this folder
