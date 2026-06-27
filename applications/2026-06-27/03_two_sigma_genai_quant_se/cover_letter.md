# Cover Letter — Two Sigma, Quantitative Software Engineer (Generative AI, News Eng)

Dear Two Sigma Hiring Team,

I'm applying for the Quantitative Software Engineer — Generative AI role on
the News Engineering team. The pairing of NLP/LLM work with a research-grade
quant environment is rare, and it's exactly the kind of place where the
combination of evaluation rigor and Bayesian inference I've built up actually
matters.

The most relevant work is a 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) I
built for harm classification: 12,500 response pairs auto-graded at **96.8%
accuracy and ROC-AUC 0.9923**, running at 850 samples/hr for $0.018/sample
(a 340x cost reduction). The substrate transfers directly to news-signal
extraction — circuit-breakered async API integration, MLflow lineage for every
run, and a stacking classifier over 47 engineered features that turn raw LLM
output into something you can hedge against. On top of that I fit a PyMC
Bayesian hierarchical model that produces 95% HDI intervals per judge,
because in any setting where a downstream decision has real cost, the right
output is a distribution, not a point.

Closely related: a bias-detection study ran 67,500 LLM ratings over 4,500
passages and 2.5M tokens, holding Krippendorff's alpha at 0.84 and finding
statistically significant publisher-level effects (Friedman chi-squared =
42.73, p < 0.001) via partial pooling (R-hat < 1.01). That's a workable
template for "extract a signal from heterogeneous text sources and quantify
how confident we should be." On the engineering side, a clinical-grade
classifier benchmarked 8 algorithms under nested cross-validation and shipped
the winner (99.12% accuracy, 100% precision, ROC-AUC 0.9987) behind a FastAPI
service at <100ms p95 — model selection by evidence, productionized cleanly.

I'd want to bring three things to News Engineering: an instinct for building
LLM pipelines that fail loudly rather than silently, a Bayesian-first habit
when the alternative is point estimates that don't survive contact with
markets, and a willingness to chase down throughput / latency / cost until
the system is actually useful. NYC-based, 2026 start once I finish my MS in
Applied Statistics at RIT.

Best,
Derek Lankeaux
