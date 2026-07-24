# Cover Letter — HealthTech / Clinical ML Engineer

**Placeholders to replace:** `[Company]`, `[Team/Product]`, `[JD hook]`.

---

Dear [Company] Hiring Team,

I'm applying for the [Team/Product] ML Engineer role. What drew me in was
[JD hook — e.g., "explainable clinical decision support at production
latency"]. That's the exact stack I've been shipping.

My **Clinical-Grade Breast Cancer Classification System** is an 8-algorithm
benchmark study (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting)
whose winning ensemble reaches **99.12% accuracy — above the 90-95% band for
human experts** — with **100% precision (zero false positives)**, 98.59%
recall, and ROC-AUC 0.9987. The pipeline is honest about the harder parts:
VIF-based multicollinearity screening, SMOTE for class balance, RFE for
feature selection, calibration, and SHAP for clinical-transparency
explanations. It ships behind a FastAPI service with **<100ms p95 latency** and
MLflow model-registry versioning, with a model card and fairness audit written
to **IEEE 2830-2025** requirements — the artifacts a compliance reviewer
actually asks for.

Two adjacent projects show the same MLOps rigor transfers. My **AI Safety
Red-Team Framework** hits 96.8% accuracy on 12,500 harm-detection pairs at
$0.018/sample with Krippendorff's alpha = 0.81. My **LLM Bias Detection**
system evaluated 67,500 ratings under a Bayesian hierarchical model with
R-hat < 1.01 and p < 0.001 significance — evidence that I sweat both the
statistics and the plumbing.

I'd like to bring that clinical-ML rigor — explainability, calibration,
SHAP-audited model cards, and low-latency serving — to [Team/Product]. Reports
and code linked in the portfolio.

Thanks,
Derek Lankeaux
