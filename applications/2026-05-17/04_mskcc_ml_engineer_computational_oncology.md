# 04 — Memorial Sloan Kettering · ML Engineer, Computational Oncology

**Location:** New York City (on-site)
**Tier:** D — NYC Healthcare
**Lead project:** Clinical-Grade Breast Cancer ML Classification
**Supporting:** AI Safety Red-Team (audit / IEEE 2830-2025 rigor), LLM Bias Detection (Bayesian uncertainty)
**JD source:** careers.mskcc.org — filter "Machine Learning" / "Computational Oncology"
**Resume version to send:** `resume_v_healthcare.pdf`

---

## Tailored Projects section

### Clinical-Grade Breast Cancer ML Classification — *Independent Research, 2025*
- Built clinical-grade classifier reaching **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** — above the 90-95% range typically cited for human expert reads
- Designed preprocessing pipeline for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation, RFE feature selection
- Implemented SHAP-based per-prediction explanations to satisfy **IEEE 2830-2025** transparency requirements for clinical decision support
- Benchmarked 8 algorithms under nested CV (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2); deployed winning ensemble as a containerized FastAPI service with MLflow registry at <100ms p95
- Full audit trail, fairness audit, and model card aligned with ISO/IEC 23894:2025 and EU AI Act

### AI Safety Red-Team Evaluation — *Jan 2026*
- Same rigor applied to GenAI: 3-LLM ensemble + 47-feature meta-classifier at 96.8% accuracy / ROC-AUC 0.9923 on a 12,500-pair benchmark; IEEE 2830-2025 audit pipeline

### LLM Ensemble Bias Detection — *2025*
- PyMC Bayesian hierarchical inference (R-hat < 1.01, ESS > 1000); 95% HDI credible intervals — the right uncertainty quantification for clinical-grade claims

---

## Cover letter

> Dear MSKCC Computational Oncology team,
>
> The work most relevant to MSK is a clinical-grade breast-cancer classifier I shipped this year. I benchmarked 8 algorithms end-to-end — Random Forest through stacking ensembles — and landed at **99.12% accuracy with 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987**, comfortably above the 90-95% range typically cited for human expert reads. Just as important for clinical deployment: the pipeline ships with SHAP explanations per prediction, VIF-pruned features, SMOTE-balanced training, and a FastAPI service under 100ms p95, all aligned with **IEEE 2830-2025** transparency standards and the EU AI Act.
>
> Two other projects show that this is a habit, not a one-off. My LLM-ensemble bias-detection study fit a PyMC partial-pooling hierarchical model (R-hat < 1.01, 95% HDI per group) over 67,500 ratings — the kind of uncertainty quantification clinical claims actually require. My AI Safety Red-Team eval framework hits 96.8% accuracy on a 12,500-pair benchmark with full IEEE 2830-2025 audit trail. The connective tissue is the same: pick the right model by evidence, quantify uncertainty honestly, ship with explainability and a real audit trail.
>
> I'd love to apply this to MSK's computational oncology work — diagnostic, prognostic, or treatment-response modeling. I'm based in / available for New York City for on-site, targeting a 2026 start after my MS in Applied Statistics wraps at RIT.
>
> Best,
> Derek Lankeaux

---

## JD-keyword echo plan
- Phrase 1: ________________________
- Phrase 2: ________________________
- Phrase 3: ________________________
