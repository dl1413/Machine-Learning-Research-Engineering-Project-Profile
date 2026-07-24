# Cover Letter — Data Scientist (Bayesian / Experimentation)

**Placeholders to replace:** `[Company]`, `[Team/Product]`, `[JD hook]`.

---

Dear [Company] Hiring Team,

I'm applying for the [Team/Product] Data Scientist role. Your posting
mentioned [JD hook — e.g., "moving experimentation to a Bayesian
decision framework"], which is the same problem I've been solving in a
different domain.

In my **LLM Ensemble Textbook Bias Detection** project, the point estimates
were the easy part — the interesting question was how confident I should be in
each publisher's bias score. I built a **Bayesian hierarchical model with
partial pooling** across 5 publishers and 4,500 passages, verified convergence
(R-hat < 1.01, healthy ESS), and reported publisher-level 95% HDIs. The
Friedman test then surfaced statistically significant bias (chi-squared = 42.73,
**p < 0.001**) in 3 of 5 publishers — a signal that, without partial pooling,
was buried under between-passage noise. The full pipeline runs at 67,500
ratings with production API scaffolding: circuit breakers, exponential backoff,
MLflow tracking.

Two other projects show the same instincts. My **AI Safety Red-Team Framework**
hits 96.8% accuracy at $0.018/sample with Krippendorff's alpha = 0.81 —
industrial-grade IRR reporting on top of a Bayesian multi-model risk analysis.
My **Clinical-Grade Classifier** ships at 99.12% accuracy / zero false
positives, with calibration (Platt, isotonic), threshold tuning, and SHAP
explainability — the same "decision under uncertainty" toolkit applied to a
binary clinical outcome.

I'd like to bring that mix — Bayesian hierarchical modeling, calibration, IRR,
and A/B design tightly scoped to the business question — to [Team/Product].
Reports and code in the portfolio.

Best,
Derek Lankeaux
