# Application 2 — Arlo, Applied AI Engineer (LLM Safety / Red-Teaming)

- **Date:** 2026-06-16
- **Location:** New York, NY
- **Source:** LinkedIn / company careers
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting:** LLM Bias Detection

## Resume bullet order (top of Projects section)

1. AI Safety Red-Team Evaluation (lead) — 96.8% accuracy, 340x cost reduction, Krippendorff alpha = 0.81 across GPT-4o / Claude-3.5 / Llama-3.2.
2. LLM Ensemble Bias Detection — 67,500 LLM ratings, async API integration with circuit breakers, MLflow lineage.
3. Breast Cancer Classification — 99.12% accuracy, FastAPI <100ms p95, SHAP explanations.

## Cover letter

Dear Arlo team,

Arlo's Applied AI Engineer role asks for hands-on LLM safety, red-teaming, and evaluation framework experience — that's exactly what I shipped last quarter. My AI Safety Red-Team Evaluation framework runs a 3-model LLM ensemble (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) as red-team judges across 6 harm categories (dangerous info, hate, deception, privacy, illegal activity, self-harm), trains a stacking classifier on 47 jailbreak / refusal-evasion / policy-violation features, and reaches 96.8% accuracy and ROC-AUC 0.9923 on a 12,500-pair benchmark. The throughput is 850 samples/hr at $0.018/sample — a 340x cost reduction versus human annotation — with Krippendorff's alpha = 0.81 holding the ensemble together. The infra side is production-grade: async API integration, circuit breakers, exponential backoff, MLflow tracking, and an adversarial taxonomy aligned to MITRE ATLAS.

Two things I'd bring to Arlo beyond the headline numbers: (1) I quantify judge disagreement with a PyMC Bayesian hierarchical model (95% HDI per model family), so you can tell which guardrail blind spots are real versus noise; and (2) every artifact is IEEE 2830-2025-compliant with SHAP explanations and full audit trails, which matters as soon as Arlo's customers sit in regulated industries.

I'm NYC-available and finishing my MS in Applied Statistics at RIT for a 2026 start.

Best,
Derek Lankeaux

## JD keywords to echo

`LLM safety`, `red-teaming`, `evaluation framework`, `prompt injection`, `jailbreak`, `guardrails`, `production`, `MLOps`, `audit`
