# Application 4 — Memorial Sloan Kettering · Computational Oncology ML Engineer

| Field | Value |
|---|---|
| Date | 2026-05-25 (Mon) |
| Company | Memorial Sloan Kettering Cancer Center |
| Role | Machine Learning Engineer, Computational Oncology |
| Role family | Healthcare / Clinical ML |
| Location | NYC (on-site) |
| Source | careers.mskcc.org |
| Lead project | Project 3 — Clinical-Grade Breast Cancer ML Classification |
| Supporting | Project 1 — Red-Team Eval (rigor + audit standards), Project 2 — Bayesian uncertainty quantification |
| Resume version | resume_v3_clinical.pdf |
| Cover letter | Yes |
| Referral | No |

---

## Tailored resume bullets

**Clinical-Grade Breast Cancer ML Classification System** — *Independent Research, Jan 2026*
- Built clinical-grade classifier exceeding the 90-95% human-expert range — **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987**
- Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified nested cross-validation, RFE feature selection
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Deployed as containerized FastAPI service with MLflow model registry; <100ms p95 latency, audit trails, model card

**Supporting research depth:**
- AI Safety Red-Team Eval: 12,500-pair benchmark, alpha = 0.81, IEEE 2830-2025 audit pipeline (same compliance standard clinical AI requires)
- LLM Bias Detection: PyMC Bayesian hierarchy with 95% HDI credible intervals, R-hat < 1.01 — uncertainty quantification habits I'd carry into clinical prediction

## Cover letter

Dear MSK Hiring Team,

The work most relevant to the Computational Oncology team is a clinical-grade
breast-cancer classifier I shipped in January. I benchmarked 8 algorithms
end-to-end — Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting,
plus two more — and landed at **99.12% accuracy with 100% precision (zero
false positives)**, 98.59% recall, and ROC-AUC 0.9987, comfortably above the
90-95% range typically cited for expert reads on this benchmark. Just as
important for clinical deployment: the pipeline ships with SHAP explanations
per prediction, VIF-pruned features, SMOTE-balanced training, and a FastAPI
service under 100ms p95, all aligned with **IEEE 2830-2025** transparency
standards and prepared for the audit trail a hospital ML system needs.

Two adjacent projects sharpen the case. An AI Safety Red-Team evaluation
framework I built shows the same compliance discipline at scale (alpha =
0.81 across LLM judges, 96.8% accuracy on 12,500 pairs, full provenance
trails). And a PyMC Bayesian hierarchical bias-detection study (MCMC R-hat
< 1.01, 95% HDI credible intervals) is exactly the uncertainty-quantification
habit I'd want around any oncology prediction — point estimates aren't enough
when a clinician is downstream.

I'm finishing my MS in Applied Statistics at RIT (2026), based in NYC and
ready to start on-site. I'd love to apply the same rigor to MSK's
computational-oncology problems.

Best,
Derek Lankeaux

## JD keyword echoes

- "clinical"
- "diagnostic" / "decision support"
- "explainability" / "SHAP"
- "model deployment"
- "cross-validation"
- "fairness audit"

## Quality checklist

- [x] Lead project re-ordered (Breast Cancer top)
- [x] Metric hook (99.12% / 100% precision)
- [x] 3+ JD phrases verbatim
- [x] IEEE 2830-2025 and clinical-context language emphasized
- [x] NYC on-site availability stated
- [x] Salary: "open, targeting market for NYC academic medical center"
- [x] Work auth: US authorized
