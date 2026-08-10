# Cover Letter — GenAI Applied Scientist (Enterprise)

**Placeholders to replace:** `[Company]`, `[Team/Product]`, `[JD hook]`.

---

Dear [Company] Hiring Team,

I'm applying for the [Team/Product] GenAI role. Your team's focus on
[JD hook — e.g., "content-quality signals at production scale"] is a good
match for what I've been building.

My **LLM Ensemble Textbook Bias Detection** system processes **67,500 ratings
across 4,500 passages (2.5M tokens)** through GPT-4o, Claude-3.5, and Llama-3.2.
The engineering side is production-grade: circuit breakers, exponential
backoff, MLflow experiment tracking, FastAPI serving. The statistics side is
what makes the ratings actually usable — a **Bayesian hierarchical model with
partial pooling and MCMC convergence (R-hat < 1.01)** gives publisher-level
credible intervals rather than point estimates, and the Friedman test surfaced
statistically significant bias (chi-squared = 42.73, **p < 0.001**) in 3 of 5
publishers with 92% pairwise LLM correlation and Krippendorff's alpha = 0.84.

The same evaluation instincts show up in adjacent work. My **AI Safety
Red-Team Framework** hits 96.8% accuracy at $0.018/sample (340x cheaper than
human labelers) with alpha = 0.81, and my **Clinical-Grade Classifier** ships
at 99.12% accuracy with <100ms p95 FastAPI serving and SHAP model cards. The
common thread: ensemble outputs, quantified uncertainty, explainable decisions
in production.

I'd like to bring that mix — LLM ensembles, Bayesian quantification, and the
FastAPI/MLflow scaffolding that keeps it reliable — to [Team/Product]. Reports
and portfolio linked below; happy to dig into any of it.

Thanks,
Derek Lankeaux
