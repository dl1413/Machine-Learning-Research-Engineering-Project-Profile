# Cover Letter — Patronus AI, Machine Learning Engineer (Remote)

Dear Patronus AI Hiring Team,

I'm applying for the Machine Learning Engineer role. I'm interested in
Patronus because I've spent the last six months building exactly the kind of
multi-LLM evaluation infrastructure your product abstracts — hallucination
checks, adversarial probes, ensemble judges, the works — and I'd rather build
it as a product than rebuild it once per company.

The most directly relevant work is an AI Safety Red-Team Evaluation framework
I shipped this year. It's a 3-LLM eval harness (GPT-4o, Claude-3.5,
Llama-3.2) that auto-grades 12,500 response pairs at **96.8% accuracy, ROC-AUC
0.9923, and 850 samples/hr** — with circuit breakers, async batching, MLflow
tracking, and exponential backoff baked in so the harness doesn't lie when
APIs flake. Cost per sample landed at **$0.018, a 340x reduction versus
human review**. The interesting part for Patronus is the stacking layer: an
XGBoost meta-classifier over 47 harm-signal features that reconciles
disagreement between the three judges, and a PyMC Bayesian hierarchical model
that surfaces per-model blind spots via 95% HDI risk intervals. That's roughly
the shape of the failure-mode triangulation your platform sells.

Closely related: a bias-detection pipeline that ran 67,500 LLM ratings over
4,500 passages and 2.5M tokens, holding Krippendorff's alpha at 0.84 and
surfacing publisher-level bias at p < 0.001. That's the eval rubric +
reliability scaffolding pattern, just pointed at content fairness instead of
harm. And a clinical-grade classifier benchmarked 8 algorithms and shipped
the winner (99.12% accuracy, 100% precision, ROC-AUC 0.9987) behind a FastAPI
service at <100ms p95 — model selection by evidence, productionized cleanly.

What I'd want to bring to Patronus: a willingness to push on adversarial test
generation (the hardest open eval problem), eval rubrics that hold their alpha
under model drift, and a Bayesian-first habit so the platform's outputs are
calibrated probabilities, not raw scores. Remote-friendly, 2026 start once I
wrap my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux
