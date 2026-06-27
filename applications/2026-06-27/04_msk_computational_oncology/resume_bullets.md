# Resume bullets — MSK, ML Scientist / Engineer (Computational Oncology)

Lead with Project 3 (Healthcare version). Supporting: Project 2 (Bayesian),
Project 1 (rigor + production).

## Clinical-Grade Breast Cancer ML Classification (LEAD)

- Built clinical-grade ensemble classifier exceeding the 90-95% human-expert range: **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987**
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) under nested cross-validation
- Applied VIF-based multicollinearity pruning, SMOTE class-balancing, RFE feature selection — recall up to 98.59% while holding precision at 100%
- Shipped SHAP per-prediction explanations to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Deployed behind containerized FastAPI service with MLflow model registry; **<100ms p95 latency**, reproducible from a single clone

## LLM Ensemble Bias Detection — Bayesian rigor (Supporting)

- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; MCMC convergence R-hat < 1.01, ESS > 1000
- 95% HDI credible intervals per group — defensible "this group is biased" claims rather than point estimates
- Friedman omnibus (chi-squared = 42.73, p < 0.001) plus Nemenyi post-hocs localized the effect
- Pattern transfers directly to multi-site clinical data (partial pooling shares strength without erasing site heterogeneity)

## AI Safety Red-Team Evaluation — auditability (Supporting)

- 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs at **96.8% accuracy, ROC-AUC 0.9923**
- Krippendorff's alpha = 0.81; IEEE 2830-2025-compliant audit pipeline with full provenance
- Template for LLM-assisted chart review / report triage with calibrated uncertainty

## JD keyword echo
clinical ML, healthcare AI, computational oncology, EHR, clinical decision
support, model explainability, SHAP, calibration, fairness audit, HPC,
multi-site, Bayesian inference, partial pooling, IEEE 2830-2025, reproducibility.
