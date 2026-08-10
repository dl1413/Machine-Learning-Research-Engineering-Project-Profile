# Cover Letter — Dataminr (Senior Research Scientist, NLP/LLM/GenAI)

Dear Dataminr Research Team,

Real-time event detection from noisy multilingual signal is an ensemble problem — no single model catches every category, and every classifier decision is a threshold on a calibrated probability. That framing is exactly what I have built three times over the last year.

My **AI Safety Red-Team** framework processes 12,500 model-response pairs at 850 samples/hour through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble, with 80K+ API calls managed by circuit breakers and exponential backoff. Six harm categories are classified by a Stacking Classifier at 96.8% accuracy (97.2% precision, ROC-AUC 0.9923), and the ensemble stage holds Krippendorff's α = 0.81 — the same disagreement-adjudication problem Dataminr must solve every second across languages and sources. Cost dropped 340× versus a human-only baseline.

My **Breast Cancer** classifier is a tighter version of the calibration and threshold work an alerting system needs. Platt scaling drove Expected Calibration Error from 0.0312 to 0.0089 (71.5% reduction), and I built context-adaptive thresholds: 100% sensitivity at τ = 0.31 for high-recall screening, higher-precision thresholds for confirmatory paths. FastAPI deployment held <100ms p95 latency with MLflow-tracked drift monitoring.

My **LLM Bias Detection** work is where I demonstrated that ensemble disagreement can be modeled, not just averaged: PyMC hierarchical partial pooling with MCMC R-hat < 1.01, per-publisher 95% HDI, Friedman χ² = 42.73 (p < 0.001), and Spearman correlations up to 0.74 across publishers revealing latent editorial structure — the same kind of source-relationship graph you would want on public-signal feeds.

I have an Applied Statistics MS (RIT, 2026), am authorized to work in the US, and am based in the NYC-metro-accessible band with full flexibility for hybrid. I would welcome a conversation about Dataminr's Research or Applied Science teams.

Sincerely,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) • [GitHub](https://github.com/dl1413) • [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
