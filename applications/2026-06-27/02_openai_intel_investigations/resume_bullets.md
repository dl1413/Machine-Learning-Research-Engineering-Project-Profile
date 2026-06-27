# Resume bullets — OpenAI, Intelligence & Investigations / T&S Engineer

Lead with Project 1 (T&S/Red-Team version). Supporting: Projects 2 + 3.

## AI Safety Red-Team Evaluation Framework (LEAD)

- Built dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-classifying 12,500 AI response pairs across 6 harm categories at **96.8% accuracy, ROC-AUC 0.9923**
- Cut cost-per-judgment **340x** ($6.12 → $0.018) while holding inter-rater reliability at Krippendorff's alpha = 0.81
- Engineered 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Quantified per-judge disagreement via PyMC Bayesian hierarchical model with 95% HDI — actionable risk intervals, not raw scores
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explanations and full provenance trail

## LLM Ensemble Bias Detection (Supporting — content policy / triangulation)

- Ran 67,500 LLM ratings over 4,500 textbook passages (2.5M tokens); 92% pairwise inter-LLM correlation
- Surfaced statistically significant publisher-level bias (Friedman chi-squared = 42.73, p < 0.001) in 3/5 publishers via PyMC partial-pooling hierarchical model (R-hat < 1.01)
- Async API layer with circuit breakers and exponential backoff sustained the workload without intervention

## Clinical-Grade Breast Cancer Classification (Supporting — high-stakes decisions)

- 99.12% accuracy, **100% precision (zero false positives)**, 98.59% recall, ROC-AUC 0.9987 on held-out test
- 8-algorithm benchmark with nested CV; SHAP explanations per prediction for auditability
- FastAPI service with MLflow model registry; <100ms p95

## JD keyword echo
trust & safety, integrity, abuse detection, content policy, red-team,
investigations, harm classification, LLM safety, jailbreak, RLHF, eval
infrastructure, Python, SQL, ML pipeline, calibration, false positives.
