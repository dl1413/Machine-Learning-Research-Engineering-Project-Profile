# Cover Letter Template — ML Research Engineer / Applied Research Scientist

**Best fit for:** Frontier labs, AI research orgs, foundation-model evaluation teams (Anthropic, OpenAI, Google DeepMind, Cohere, Scale AI, AI2, Hugging Face, NYU CDS-adjacent labs).

**Anchor projects (in priority order):** AI Safety Red-Team Evaluation → LLM Bias Detection → Breast Cancer Classification (as proof of statistical rigor).

---

Dear {{Hiring Manager Name | Hiring Team}},

I'm applying for the **{{Role Title}}** position at **{{Company}}**. Your work on {{specific paper / product / model / eval — 1 concrete reference}} is the reason I'm writing: it's adjacent to the problems I've been building solutions for over the past year.

I'm a Machine Learning Research Engineer finishing my MS in Applied Statistics at RIT (Bayesian methods, experimental design). My independent research over the past year has focused on building production-grade LLM evaluation systems that hold up under statistical scrutiny:

- **AI Safety Red-Team Evaluation Framework** — Engineered a dual-stage ensemble (GPT-4o + Claude-3.5 + Llama-3.2 → Stacking Classifier) that evaluates 12,500 AI response pairs across 6 harm categories with **96.8% accuracy** and Krippendorff's α = 0.81. The system delivers a **340× cost reduction** vs. human annotation ($0.018/sample) at 850 samples/hour, with a MITRE ATLAS–aligned attack taxonomy and Bayesian hierarchical risk modeling. IEEE 2830-2025 compliant.

- **LLM Ensemble Bias Detection** — Processed 67,500 bias ratings across 4,500 textbook passages (2.5M tokens). Achieved α = 0.84 inter-rater reliability with PyMC hierarchical partial-pooling models (R-hat < 1.01) and identified statistically significant publisher-level bias (Friedman χ² = 42.73, p < 0.001).

- **Clinical-Grade Breast Cancer Classifier** — 99.12% accuracy with 100% precision and ROC-AUC 0.9987 across an 8-algorithm benchmark, Optuna TPE hyperparameter search, and Platt calibration (ECE 0.0089). Deployed via FastAPI at <100ms p95 latency.

Three things I'd bring to {{Company}}:

1. **Evaluation rigor at production scale** — I've shipped ensembles that maintain α ≥ 0.81 reliability while processing 80K+ annotations, with circuit breakers, MLflow tracking, and SHAP audit trails baked in from day one.
2. **Statistical maturity** — Bayesian uncertainty quantification, MCMC diagnostics, and multiple-testing correction are part of my default workflow, not bolted on. My reports document 95% HDI intervals, effect sizes, and power analyses.
3. **End-to-end ownership** — From dataset construction and feature engineering through deployment and monitoring, I've taken every one of these projects from blank page to published technical report.

I'd welcome the chance to talk through how this maps to {{specific team / problem mentioned in JD}}. Portfolio and reports are at [github.com/dl1413](https://github.com/dl1413) and [dl1413.github.io/LLM-Portfolio](https://dl1413.github.io/LLM-Portfolio/).

Thank you for your time.

Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · Authorized to work in the US · Remote / NYC

---

### Customization notes
- Line 1: Always replace `{{specific paper / product / model / eval}}` with one real, named artifact from the company. If you can't find one in 5 minutes of searching, skip this whole company.
- Project order: Lead with **AI Safety Red-Team** for frontier-lab safety/alignment roles; lead with **Bias Detection** for fairness/responsible-AI roles; lead with **Breast Cancer** only if the JD names healthcare or classical ML.
- Length target: 280–350 words after filling placeholders. Cut the third project bullet first if the JD is short.
