# Oscar Health — Data Scientist I, Medical Cost Analytics

**Location:** New York, NY (hybrid)
**Apply:** https://jobs.technyc.org/companies/oscar-health/jobs/45698347-data-scientist-i-medical-cost-analytics
**Lead project:** Clinical-Grade Breast Cancer ML Classification
**Supporting projects:** LLM Bias Detection (Bayesian hierarchical modeling for population-level effects), AI Safety Red-Team (LLM-assisted analytics at scale)

---

Dear Oscar Health team,

Medical cost analytics sits at the intersection of clinical rigor, calibrated risk quantification, and stakeholder-facing communication — the three things I've spent my MS in Applied Statistics learning to do together. I'd like to apply for the Data Scientist I role on the Medical Cost Analytics team.

The best evidence I can point to is a clinical-grade ML classification system I built and published in April 2026 on the Wisconsin Breast Cancer benchmark. I ran an 8-algorithm ensemble comparison (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting) with Bayesian hyperparameter tuning via Optuna's TPE, arriving at an AdaBoost model with **99.12% accuracy, 100% precision, 98.59% recall, and ROC-AUC 0.9987**. The preprocessing pipeline handled the things that matter for real clinical/actuarial data: VIF-based multicollinearity screening, SMOTE for class imbalance, and RFE feature selection with cross-validated stability checks (98.46% ± 1.12% across folds).

More important for Medical Cost work: I paired the model with Platt calibration that cut expected calibration error 71.5% (0.0312 → 0.0089), then implemented context-adaptive thresholds — for example, a 0.31 threshold that hits 100% sensitivity for mass screening use cases. That's the same "calibrated probabilities → decision policy" mapping actuarial teams need when a predicted cost is going into a rate filing or a clinical program. I documented the whole system with SHAP-based explanations, model cards, and audit trails aligned to IEEE 2830-2025 (Transparent ML) — because in regulated domains, a model that can't explain itself is a model that can't ship.

I also bring modern GenAI depth Oscar increasingly needs: I built a multi-LLM Bayesian bias-detection framework processing 2.5M tokens across 67,500 ratings with Krippendorff's α = 0.84 and PyMC hierarchical inference (R-hat < 1.01) — directly applicable to LLM-assisted claims triage, prior-auth notes analysis, or member-communication QA.

I'm hybrid-friendly for the NYC office and available to start in 2026. Would love a first conversation.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
