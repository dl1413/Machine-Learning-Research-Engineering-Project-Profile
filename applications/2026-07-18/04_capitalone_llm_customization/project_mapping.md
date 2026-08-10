# Project → JD Mapping — Capital One LLM Customization

## LLM Ensemble Bias Detection (PRIMARY MATCH)

| Their ask | My evidence |
|-----------|-------------|
| Multi-model LLM work at production scale | GPT-4o + Claude-3.5 + Llama-3.2 via LangChain; 92% pairwise correlation; 2.5M tokens processed |
| Reusable eval infrastructure | MLflow experiment tracking, circuit breakers, exponential backoff — plug-and-play across new prompt sets |
| Statistical rigor for ship / no-ship decisions | PyMC hierarchical model, partial pooling, R-hat < 1.01, 95% HDI, Friedman χ² = 42.73 (p < 0.001) |
| Uncertainty-aware routing | 12.3% high-uncertainty items flagged for expert review rather than auto-decided |

## AI Safety Red-Team Framework (SECONDARY — LLM-as-judge + business case)

| Their ask | My evidence |
|-----------|-------------|
| LLM-as-judge fluency | Ensemble annotation → 47-feature Stacking Classifier at 96.8% accuracy (97.2% precision, 96.1% recall) |
| Cost-to-quality tradeoff | $0.018/sample vs $6.12 human ⇒ 340× cost reduction, α = 0.81 preserved |
| Prompt iteration + eval design | 8-vector attack taxonomy, dual-filter defense measurement (21.8% → 4.8% harm rate) |
| Regulated-org governance | SHAP audit trails, model cards, IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act aligned |

## Breast Cancer Classification (TERTIARY — production + calibration)

| Their ask | My evidence |
|-----------|-------------|
| Production ML hygiene | MLflow registry, FastAPI < 100 ms p95, calibrated probabilities (ECE 0.0089) |
| Threshold policies for decisions | Context-adaptive thresholds — 100% sensitivity at 0.31 for screening |
| Bayesian hyperparameter search | Optuna TPE, 5× fewer trials than grid search |

## Short-answer prompts (drop-in)

- **"How would you decide which fine-tuned checkpoint to ship?"** → Hierarchical Bayesian eval, 95% HDI comparison, α ≥ 0.80 reliability floor, non-parametric corroboration, cost-per-eval tracked so the decision is repeatable.
- **"How would you make LLM evals cheaper without losing signal?"** → 340× cost reduction case study with preserved α = 0.81 IRR, plus uncertainty flagging so the humans stay in the loop where it matters (12.3% of items).
- **"Business impact of your work?"** → Cost reduction + audit-grade governance + 78% harm-rate reduction on the defense side — three concrete deltas, not accuracy alone.
