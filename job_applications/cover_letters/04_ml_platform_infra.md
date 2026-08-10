# Cover Letter Template — ML Platform / MLOps / Infra Engineer

**Best fit for:** ML platform teams (Databricks, Weights & Biases, MLflow/Databricks, Hugging Face infra, Ramp ML platform, Datadog ML monitoring, Anyscale, Modal, Replicate), MLOps-heavy roles at enterprise companies.

**Anchor:** The infrastructure across all 3 projects — pipelines, MLflow, FastAPI, circuit breakers. Less about model accuracy, more about reliability + reproducibility.

---

Dear {{Hiring Manager Name | Platform Team}},

I'm writing to apply for the **{{Role Title}}** position at **{{Company}}**. {{Specific reference — open-source repo, blog post about your platform's design, recent product launch}}.

The throughline of my last year of work has been building ML systems that hold their shape outside the notebook. Across three independent research projects I've put roughly equal effort into the model and into the infrastructure that runs it:

- **80,000+ LLM annotations processed** across GPT-4o, Claude-3.5, and Llama-3.2 with circuit breakers, exponential backoff, adaptive rate limiting, structured logging (structlog), and full MLflow experiment tracking. Sustained 850 samples/hour throughput without manual intervention.

- **FastAPI serving at <100ms p95 latency** with calibrated probabilities (Platt scaling, ECE 0.0089), SHAP explanations exposed as endpoints, and MLflow model registry for versioning and rollback.

- **Reproducibility-first workflow**: fixed seeds, version-pinned environments, model cards per artifact, automated PDF report generation (LaTeX), and explicit IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act compliance checklists.

- **Ensemble pipelines** that span 8+ algorithms (sklearn, XGBoost, LightGBM, AdaBoost, Stacking) and 3 LLM providers, with Optuna TPE for hyperparameter search (converged in 45 trials vs. 240 for grid).

I'm bringing two things I think matter for a platform role:

1. **Empathy for the model-developer side** — I've been the person filing the bugs against MLflow, hitting the rate-limit cliffs, and writing the circuit breakers. I know which pain points are real and which are tooling smells.
2. **Statistical literacy in the platform layer** — Drift detection, calibration monitoring, A/B significance — these are statistics problems, and getting them wrong silently is the most expensive failure mode for an ML platform. I've published on inter-rater reliability and Bayesian inference, so I won't ship a half-baked test.

I'd love to talk about how this maps to {{specific platform component / open problem from JD}}. Code and reports: [github.com/dl1413](https://github.com/dl1413).

Thank you,

Derek Lankeaux
MS Applied Statistics, RIT (2026) · [LinkedIn](https://linkedin.com/in/derek-lankeaux) · US work auth · Remote / NYC

---

### Customization notes
- This is the "I know your pain because I've been your user" angle — works best at platform companies whose product is used by ML engineers.
- If the company is closer to traditional SRE/infra than ML platform, swap point 2 for "Production reliability instincts — graceful degradation, idempotent retries, observability."
- Don't lead with model accuracy numbers — they're a tax on the reader for a platform role.
