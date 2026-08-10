# Project → JD Mapping — Dataminr Senior Research Scientist

## AI Safety Red-Team Framework (PRIMARY MATCH)

| Their ask | My evidence |
|-----------|-------------|
| High-precision signal detection | 97.2% precision, 96.1% recall, ROC-AUC 0.9923 — same statistical shape as low-FPR real-time signal detection |
| Research → production | Ensemble LLM annotation → engineered features → Stacking Classifier, MLflow-tracked, 850 samples/hr |
| Adversarial robustness | 8-vector attack taxonomy; multi-turn escalation identified as highest-risk (31.8%); dual-filter defense measured |
| Cost realism at scale | $0.018/sample, 340× reduction, α = 0.81 preserved |

## LLM Ensemble Bias Detection (SECONDARY — publication-grade research)

| Their ask | My evidence |
|-----------|-------------|
| Bayesian methodology | PyMC partial pooling, R-hat < 1.01, 95% HDI |
| Non-parametric corroboration | Friedman χ² = 42.73, p < 0.001 |
| Structural analysis | Spearman correlation matrix, ρ up to 0.74 across publishers |
| External communication | Publication-grade technical report ready to send as a writing sample |

## Breast Cancer Classification (TERTIARY — calibration + latency)

| Their ask | My evidence |
|-----------|-------------|
| Calibrated confidence per event | Platt scaling reduces ECE 71.5% (0.0312 → 0.0089) |
| Latency for real-time systems | FastAPI < 100 ms p95, MLflow registry |
| Feature-level explainability | SHAP for stakeholder / customer briefings |

## Short-answer prompts (drop-in)

- **"How would you evaluate a real-time signal detector?"** → Precision at operating point + calibrated per-event confidence + drift monitoring; α ≥ 0.80 IRR floor on the labeling side.
- **"What's your bar for a research writeup?"** → 3 technical reports aligned with IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act; happy to share PDFs directly.
- **"Tell me about a research finding you had to defend."** → Multi-turn escalation as top attack vector (31.8%) — defended with both Bayesian HDIs and bootstrap CIs after reviewers pushed back on the point estimate.
