# Cover Letter — Anthropic
**Role:** Research Engineer, AI Observability
**Location:** NYC / SF / Remote-friendly (see posting)
**Source:** job-boards.greenhouse.io/anthropic/jobs/5125083008
**Anchor Projects:** AI Safety Red-Team Evaluation (primary) · LLM Ensemble Bias Detection

---

Dear Anthropic Safeguards / Observability Team,

I'm applying for the Research Engineer, AI Observability role. Anthropic's evaluation and safeguards work is the closest professional analog to what I've been building independently for the past year — automated harm-detection frameworks, LLM-as-judge ensembles, and statistical guarantees on eval reliability.

**Directly analogous work.** My AI Safety Red-Team Evaluation Framework built a **dual-stage LLM ensemble → ML classifier** across **12,500 response pairs and 6 harm categories** (dangerous info, hate, deception, privacy, illegal, self-harm). Results:

- **96.8% accuracy, 97.2% precision, 96.1% recall (Stacking Classifier, ROC-AUC 0.9923)**
- **Krippendorff's α = 0.81** across GPT-4o / Claude-3.5 / Llama-3.2 — audit-grade ensemble reliability
- **340× cost reduction** ($0.018/sample vs. $6.12 human)
- **MITRE ATLAS-aligned adversarial taxonomy** with **multi-turn escalation identified as the highest-risk vector (31.8%)**
- **Dual-filter defense: 21.8% → 4.8% observed harm rate (78% reduction)**
- **Bayesian hierarchical modeling (PyMC, 95% HDI)** for multi-model vulnerability comparison

**LLM evaluation with statistical guarantees.** My companion LLM Bias Detection project processed **67,500 ratings across 4,500 passages** with **α = 0.84**, MCMC convergence at **R-hat < 1.01**, and **Friedman χ² = 42.73 (p < 0.001)** — the kind of "is this eval difference real?" evidence Anthropic needs to move a safeguard from experimental to production.

**Engineering side.** Python 3.12+, MLflow, FastAPI, Docker; production API integration with circuit breakers and exponential backoff across **80K+ calls / 2.5M tokens**; SHAP + model cards aligned to IEEE 2830-2025.

I'd be excited to help build the eval and observability layer that lets Anthropic ship safeguards with confidence.

Best regards,
**Derek Lankeaux, MS**
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
