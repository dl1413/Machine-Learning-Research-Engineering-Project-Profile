# Cover Letter — Applied Data Scientist / LLM Evaluation (Remote)

**Role:** Technical Lead — Applied Data Scientist (LLM Evaluation focus)
**Target listing (representative):** Built In / builtin.com
**Location:** Remote (US)
**Salary band (posted):** $120K – $180K
**Source:** https://builtin.com/job/technical-lead-applied-data-scientist/7021484
**Date drafted:** 2026-07-21

---

Dear Hiring Team,

I'm applying for the Applied Data Scientist role focused on LLM evaluation and adversarial testing. The three research projects I shipped this year are what you'd get in the first 90 days: an LLM-as-judge pipeline, a Bayesian measurement framework, and a calibrated classifier serving predictions in production.

**LLM-as-judge pipeline.** *AI Safety Red-Team Evaluation* runs a GPT-4o / Claude-3.5 / Llama-3.2 ensemble scoring 12,500 responses across 6 categories at 96.8% accuracy and Krippendorff's α = 0.81 — a **340× cost reduction vs. human annotation** ($6.12 → $0.018/sample) that preserved reliability. Every organization scaling LLM eval hits the human-budget wall; this is the pattern I'd bring on day one.

**Bayesian evaluation framework.** *LLM Ensemble Textbook Bias Detection* built a PyMC hierarchical model over 67,500 ratings with partial pooling, MCMC convergence (R-hat < 1.01), and 95% HDI on every publisher-topic effect. Bootstrap CIs flagged 12.3% of passages as high-uncertainty for expert review — the routing policy that keeps eval budgets sane at scale.

**Calibrated classifiers in production.** *Breast Cancer ML Classification* (99.12% accuracy, 100% precision, ROC-AUC 0.9987) uses Platt scaling to bring ECE to 0.0089 and ships behind a FastAPI service (<100ms p95) with MLflow tracking. The methodology transfers directly to any regression-test or evaluation classifier that has to fire alerts on a live model.

**Production hygiene.** 80K+ API calls processed with circuit breakers and exponential backoff, MLflow experiment tracking, SHAP explanations, and model cards aligned with IEEE 2830-2025 and the EU AI Act. I write reports that make sense to product and compliance, not just other ML people.

I'd like the chance to bring this stack to your team as a Technical Lead.

Sincerely,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux | GitHub: https://github.com/dl1413

---

## Talking points for phone screen

- **Human-budget wall:** 340× cost reduction is a story every eval team wants to hear.
- **Regression testing:** Bayesian HDI on eval metrics beats raw t-tests on daily runs.
- **Compliance:** IEEE 2830-2025 + EU AI Act (Aug 2, 2026) is my default output.
