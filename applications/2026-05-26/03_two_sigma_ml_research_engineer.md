# 03 — Two Sigma · ML Research Engineer

- **Location:** NYC (on-site/hybrid)
- **Family:** ML Research Engineer (Finance)
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P3 Breast Cancer (ensembles + production), P2 Bias (Bayesian)
- **JD link:** https://www.twosigma.com/careers/ (ML Research Engineer)
- **Resume version:** `resume_v3_research.pdf`
- **Cover letter:** YES

## JD keywords to echo verbatim (≥3)
1. "research engineer"
2. "ensemble"
3. "Bayesian"
4. "production"
5. "feature engineering"

## Resume bullet stack

**AI Safety Red-Team Evaluation (lead — Research Engineer variant)**
- Trained stacking classifier on 47 engineered features for harm detection; 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923 on 12,500-sample benchmark
- Modeled multi-judge uncertainty via PyMC Bayesian hierarchy, producing 95% HDI risk intervals for downstream policy decisions
- Documented results in research-grade technical report; reproducible pipeline released alongside published findings

**Breast Cancer Classification (supporting — applied ML)**
- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) and shipped winning ensemble at 99.12% accuracy / ROC-AUC 0.9987
- VIF-based multicollinearity pruning, SMOTE class-balancing, and RFE feature selection lifted recall to 98.59% while holding precision at 100%
- Deployed containerized FastAPI service with MLflow model registry; p95 latency under 100ms

**LLM Textbook Bias (supporting — stats)**
- PyMC partial-pooling hierarchy, R-hat < 1.01, ESS > 1000; Friedman χ² = 42.73, p < 0.001

## Cover letter

> Dear Two Sigma research hiring team,
>
> I want to apply for the ML Research Engineer position in New York. What
> I'd bring to the team is the combination of stacked-ensemble modeling,
> Bayesian inference, and production engineering you ask for — but proved
> out on AI eval and clinical-classification problems rather than markets.
>
> The lead example: my AI Safety Red-Team Evaluation framework. I engineered
> 47 linguistic, semantic, and structural features over 12,500 model-response
> pairs and stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into an XGBoost
> meta-classifier hitting 96.8% accuracy and ROC-AUC 0.9923. I wrapped the
> judge-disagreement signal in a PyMC Bayesian hierarchical model with 95%
> HDI per rater, and shipped the eval harness at 850 samples/hr behind
> circuit breakers and MLflow tracking. Cost per sample landed at $0.018,
> a 340x cut versus human annotation.
>
> The applied-ML version of those same habits: an 8-algorithm benchmark on
> a clinical classification task — Random Forest, XGBoost, LightGBM,
> AdaBoost, stacking, voting — landing at 99.12% accuracy, 100% precision,
> ROC-AUC 0.9987, productionized as a FastAPI service under 100ms p95.
> And a third project (textbook-bias inference) carries the Bayesian thread:
> 67,500 LLM ratings, partial-pooling hierarchy with R-hat < 1.01,
> Friedman χ² = 42.73, p < 0.001 on publisher-level effects.
>
> I'm based in / available for New York City, targeting a 2026 start once
> I wrap my MS in Applied Statistics at RIT.
>
> Best,
> Derek Lankeaux
