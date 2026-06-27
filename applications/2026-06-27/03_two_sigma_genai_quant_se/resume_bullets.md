# Resume bullets — Two Sigma, Quantitative Software Engineer (Generative AI)

Lead with Project 1 (LLM Eval / Research Engineer version). Supporting: 2 + 3.

## AI Safety Red-Team Evaluation Framework (LEAD)

- Built production eval harness processing **850 samples/hr** with circuit breakers, exponential backoff, async batching, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into XGBoost meta-classifier reaching **96.8% accuracy / ROC-AUC 0.9923** on 12,500-sample benchmark
- Cut human-eval cost **340x** ($0.018/sample) while preserving Krippendorff's alpha = 0.81
- Quantified judge disagreement with PyMC Bayesian hierarchy, 95% HDI risk intervals — surfaced systematic per-model blind spots

## LLM Ensemble Bias Detection (Supporting — NLP at scale)

- Operated 3-LLM ensemble: **67,500 ratings, 4,500 passages, 2.5M tokens**; 92% pairwise inter-LLM correlation, alpha = 0.84
- Fit PyMC partial-pooling hierarchical model across publishers; MCMC convergence R-hat < 1.01, ESS > 1000
- Friedman omnibus test (chi-squared = 42.73, p < 0.001) plus Nemenyi post-hocs localized the effect to 3 of 5 publishers

## Clinical-Grade Classification (Supporting — model selection rigor)

- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) under nested CV; **99.12% accuracy, 100% precision, ROC-AUC 0.9987**
- Productionized winning ensemble behind FastAPI with MLflow registry; <100ms p95 latency

## JD keyword echo
NLP, LLM, generative AI, ML engineer, Python, news signal extraction,
PyTorch, transformers, model deployment, latency, MLflow, async, eval,
Bayesian inference, hierarchical models, ensemble methods.
