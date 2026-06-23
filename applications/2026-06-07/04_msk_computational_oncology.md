# 04 — Memorial Sloan Kettering · ML Engineer, Computational Oncology

| Field | Value |
|---|---|
| **Company** | Memorial Sloan Kettering Cancer Center |
| **Role** | Machine Learning Engineer / Research Scientist — Computational Oncology |
| **Location** | NYC (on-site, hybrid possible) |
| **Source** | careers.mskcc.org (apply direct) |
| **JD link** | _VERIFY before submit:_ https://careers.mskcc.org/ (filter: ML / Computational / Oncology / Pathology) |
| **Lead project** | P3 — Breast Cancer Classification |
| **Supporting** | P1 — Red-Team (rigor, audit-grade); P2 — LLM Bias (Bayesian uncertainty) |
| **Resume version** | `resume_v3_healthcare.pdf` |
| **Cover letter** | Yes — see below |
| **Referral** | Check RIT alumni + MS oncology authors I've cited |

## JD keyword echo (fill in after reading JD)

- [ ] clinical / diagnostic / oncology
- [ ] SHAP / interpretability / explainability
- [ ] calibration / sensitivity / specificity
- [ ] [MSK-specific: "translational", "decision support", "histopathology"]
- [ ] [MSK-specific phrase 2]

## Resume bullets (top 4 — paste into "Selected Projects")

From `APPLICATION_SNIPPETS.md` → P3 → "Healthcare / Clinical ML":

- Built clinical-grade breast-cancer classifier exceeding the 90–95% human-expert range (99.12% accuracy, 100% precision, ROC-AUC 0.9987), with zero false positives across the held-out test set
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Designed clinical tabular pipeline: VIF multicollinearity diagnostics, SMOTE class balancing, stratified cross-validation, Platt calibration (ECE 0.0089)
- Tuned context-adaptive thresholds (100% sensitivity at 0.31 for mass screening; precision-optimized for confirmatory workflows)

## Cover letter draft

Dear MSK Computational Oncology team,

The work most relevant to MSK is a clinical-grade breast-cancer
classifier I shipped this year. I benchmarked 8 algorithms end-to-end
— Random Forest through stacking ensembles — and landed at 99.12%
accuracy with 100% precision (zero false positives), 98.59% recall,
and ROC-AUC 0.9987, comfortably above the 90–95% range typically cited
for human expert reads. Just as important for clinical deployment: the
pipeline ships with SHAP explanations per prediction, VIF-pruned
features, context-adaptive thresholds (100% sensitivity for screening
vs. precision-optimized for confirmatory), Platt-calibrated probabilities
at ECE 0.0089, and a FastAPI service under 100ms p95 — all aligned
with IEEE 2830-2025 transparency standards.

Two adjacent projects show I take statistical rigor and uncertainty
seriously, which matters for any decision-support system in oncology:
a Bayesian hierarchical study (PyMC, R-hat < 1.01, 95% HDI per
unit) of 67,500 ratings, and an audit-grade LLM evaluation framework
at Krippendorff's α = 0.81 across 12,500 cases. The habit I want
to bring to MSK: pick models by evidence not fashion, quantify
uncertainty per prediction, and ship the explanations clinicians need.

I'm finishing an MS in Applied Statistics at RIT and based in /
available for NYC. Portfolio, code, and the three technical reports
are on my GitHub (dl1413). Happy to walk through any of them — the
breast-cancer classifier is probably the fastest way to see how I
think about clinical-grade modeling.

Best,
Derek Lankeaux
