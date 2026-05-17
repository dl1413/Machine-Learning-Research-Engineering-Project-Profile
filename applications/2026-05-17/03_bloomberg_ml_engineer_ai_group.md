# 03 — Bloomberg · ML Engineer, AI Group

**Location:** New York City (on-site / hybrid)
**Tier:** C — NYC Finance
**Lead project:** Clinical-Grade Breast Cancer ML Classification
**Supporting:** AI Safety Red-Team (rigor + production), LLM Bias Detection (Bayesian inference)
**JD source:** bloomberg.com/careers — filter "Machine Learning Engineer" / "AI Engineering"
**Resume version to send:** `resume_v_mle.pdf`

---

## Tailored Projects section

### Clinical-Grade ML Classification System — *Independent Research, 2025*
- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) under nested cross-validation; shipped winning ensemble at **99.12% accuracy / ROC-AUC 0.9987**
- Achieved **100% precision (zero false positives) and 98.59% recall** on the held-out test set via VIF-based multicollinearity pruning, SMOTE class balancing, and RFE feature selection
- Productionized behind a containerized FastAPI service with MLflow model registry; **p95 latency under 100ms**
- Implemented SHAP-based per-prediction explanations and a full fairness / bias audit; pipeline reproducible end-to-end

### AI Safety Red-Team Evaluation — *Jan 2026*
- 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) + XGBoost stacking classifier; 96.8% accuracy on 12,500-pair benchmark
- 850 samples/hr production harness with circuit breakers, exponential backoff, MLflow lineage — same MLOps discipline applied to LLM-as-judge

### LLM Ensemble Bias Detection — *2025*
- 67,500 ratings across 2.5M tokens; PyMC partial-pooling hierarchical model (R-hat < 1.01); Friedman test p < 0.001

---

## Cover letter

> Dear Bloomberg AI Engineering team,
>
> A recent project that captures how I work: I built a clinical-grade ensemble classifier — **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** — by benchmarking 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) under nested cross-validation, then productionized the winner behind a FastAPI service at **<100ms p95** with an MLflow model registry. The habits I want to bring to Bloomberg: pick models by evidence not fashion, ship the full pipeline (preprocessing, SHAP explanations, monitoring), and treat latency as a feature.
>
> Two adjacent projects show how the same discipline transfers to language and to inference. My AI Safety Red-Team eval framework runs a 3-LLM ensemble at 850 samples/hr / $0.018 a sample, 340x cheaper than human review at 96.8% accuracy. My LLM-ensemble bias-detection study fit a PyMC partial-pooling hierarchy (R-hat < 1.01, ESS > 1000) over 67,500 ratings, producing 95% HDI credible intervals — the right tool when a point estimate isn't defensible.
>
> Bloomberg is the place that's been doing rigorous applied ML on financial text long before it was fashionable, and I'd be excited to contribute. I'm based in / available for New York City, targeting a 2026 start after my MS in Applied Statistics wraps at RIT.
>
> Best,
> Derek Lankeaux

---

## JD-keyword echo plan
- Phrase 1: ________________________
- Phrase 2: ________________________
- Phrase 3: ________________________
