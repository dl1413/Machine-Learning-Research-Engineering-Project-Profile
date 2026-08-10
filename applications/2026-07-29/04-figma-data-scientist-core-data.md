# Figma — Data Scientist, Core Data (2026)

**Location:** New York, NY / San Francisco, CA
**Apply URL:** https://job-boards.greenhouse.io/figma/jobs/5976930004
**Team charter (per posting):** Core Data — advance the experimentation platform, build ML-based analytical systems, and measure AI-powered features through causal inference and statistical modeling.

*Degree-line note: this specific requisition is tagged "PhD (2026)." Figma often lists an adjacent MS-eligible Core Data role; before submitting, search Figma careers for "Data Scientist, Core Data" and use the MS variant if one is open. If not, apply as an MS candidate and let the recruiter route.*

---

## Cover Letter

Dear Figma Core Data Hiring Team,

I'm writing about the Data Scientist, Core Data role. Two lines in the description made me want to apply: advancing the experimentation platform, and measuring AI-powered features with causal inference. My MS in Applied Statistics from RIT is exactly that — Bayesian inference, experimental design, causal methods — and my portfolio is three end-to-end projects that put it into practice.

The one closest to Figma's AI-feature measurement work is a **multi-LLM bias-detection framework** on **4,500 textbook passages (67,500 ratings, 2.5M tokens)**. GPT-4o + Claude-3.5 + Llama-3.2 rate every passage; a **PyMC hierarchical model with partial pooling (MCMC R-hat < 1.01)** returns per-publisher **credible intervals (95% HDI)**. Pairwise LLM agreement was **92%**, the omnibus effect was significant (**Friedman χ² = 42.73, p < 0.001**), and **3/5 publishers** carried credible bias after multiple-testing correction. It's the same shape as measuring whether an AI feature moves a metric: multiple noisy judges, hierarchical structure across cohorts, a credible interval instead of a p-hacked point estimate.

On the experimentation-platform side, my **AI Safety Red-Team** framework carries the tooling story: **12,500 samples, 6 categories, 96.8% accuracy, α = 0.81, 850 samples/hour at $0.018/sample** (340× cost reduction vs. human labeling). MLflow for tracking, SHAP for interpretation, circuit breakers and exponential backoff for the API layer — the plumbing an experimentation platform needs to run a test that scales past a notebook. I've also done the classical-ML calibration work — **Breast Cancer classifier, 99.12% accuracy, Platt-calibrated ECE 0.0089, context-adaptive threshold tuning** — which is the same discipline experimentation systems need when raw scores drive decisions.

I'd welcome a conversation about the team's near-term experimentation and AI-measurement priorities, and I'm happy to work through a technical case in either language (Python and R).

Regards,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux · GitHub: https://github.com/dl1413

---

## Role-tailored resume bullets

- Measured LLM bias hierarchically across **5 publishers × 67.5K ratings** with a PyMC partial-pooling model (**R-hat < 1.01, 95% HDI, Friedman χ² = 42.73, p < 0.001**) — the credible-interval discipline an experimentation platform needs for AI-feature measurement.
- Ran **12,500-sample LLM-eval loops at $0.018/sample and 850/hr** with MLflow, SHAP, and circuit breakers — end-to-end experimentation plumbing at API scale.
