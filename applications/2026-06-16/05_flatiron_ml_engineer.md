# Application 5 — Flatiron Health, Machine Learning Engineer

- **Date:** 2026-06-16
- **Location:** New York, NY (hybrid)
- **Source:** flatiron.com/careers
- **Lead project:** Breast Cancer Classification
- **Supporting:** LLM Bias Detection (LLM-on-text extraction rigor), AI Safety Red-Team (audit infra)

## Resume bullet order

1. **Breast Cancer Classification (lead)** — 99.12% accuracy / 100% precision / ROC-AUC 0.9987 with Optuna Bayesian HPO (5x fewer trials than grid search), Platt-calibrated probabilities (ECE 0.0089), and FastAPI <100ms p95.
2. **LLM Ensemble Bias Detection** — 2.5M tokens processed across 4,500 passages with Krippendorff alpha = 0.84 — directly relevant to extracting structured signal from unstructured oncology notes / abstracts.
3. **AI Safety Red-Team Evaluation** — production eval harness, circuit breakers, audit trails — the kind of infra that meets real-world deployment scrutiny.

## Cover letter

Dear Flatiron Machine Learning team,

The most directly relevant project I'd bring to Flatiron is a clinical-grade breast-cancer classification system I shipped earlier this year: an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) tuned with Optuna's TPE sampler — converging in 5x fewer trials than grid search — and landing at 99.12% accuracy, 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987. The pipeline is production-shaped: VIF multicollinearity diagnostics, SMOTE balancing, RFE feature selection, Platt scaling (ECE reduced from 0.0312 to 0.0089), SHAP explanations per prediction, and a containerized FastAPI service at <100ms p95 — all reproducible from MLflow tracking and aligned with IEEE 2830-2025 transparency.

The reason Flatiron is interesting to me specifically is that oncology insights live as much in unstructured notes as in structured features. My LLM-ensemble bias-detection study processed 2.5M tokens across 4,500 textbook passages using GPT-4o / Claude-3.5 / Llama-3.2 with Krippendorff's alpha = 0.84 inter-rater reliability and a PyMC partial-pooling hierarchical model (R-hat < 1.01) for defensible findings. That same eval / reliability scaffolding is what NLP on clinical text needs to be trusted by oncologists and regulators.

NYC-based, finishing my MS in Applied Statistics at RIT (2026 start), work-authorized.

Best,
Derek Lankeaux

## JD keywords to echo

`real-world evidence`, `oncology`, `EHR`, `unstructured clinical data`, `production ML`, `model validation`, `MLOps`, `interpretability`, `clinical research`
