# Cover Letter — LLM Evaluation / Applied Scientist

**Placeholders to replace:** `[Company]`, `[Team/Product]`, `[JD hook]`.

---

Dear [Company] Hiring Team,

I'm applying for the [Team/Product] Applied Scientist role. Your posting
called out [JD hook — e.g., "scalable LLM-as-judge evaluation"], which is
exactly the problem I spent this year solving end-to-end.

My AI Safety Red-Team Evaluation Framework runs a dual-stage LLM ensemble
(GPT-4o + Claude-3.5 + Llama-3.2) over 12,500 response pairs across six harm
categories, hitting **96.8% accuracy** and **Krippendorff's alpha = 0.81** —
audit-grade IRR at **$0.018/sample vs. $6.12 for human annotation, a 340x cost
reduction**. The Stacking Classifier layer reaches 97.2% precision / 96.1%
recall / ROC-AUC 0.9923 at 850 samples/hour on a single production pipeline.
Uncertainty is quantified with a Bayesian hierarchical model (95% HDI, R-hat
< 1.01), and every decision is SHAP-explainable with an IEEE 2830-2025
compliant audit trail.

Two adjacent projects show the same instincts generalize. My LLM Ensemble
Textbook Bias Detection scored **67,500 ratings across 4,500 passages** with
92% pairwise correlation across frontier LLMs and surfaced statistically
significant bias in 3 of 5 publishers (Friedman chi-squared = 42.73, p < 0.001).
My Clinical-Grade Breast Cancer Classifier ships at **99.12% accuracy** with
zero false positives, served through FastAPI at <100ms p95 — the same
disciplined MLOps loop, in a different domain.

I'd like to bring that same "ensemble + IRR + explainability" playbook to
[Team/Product]. Three reports (IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act
aligned) and the code are on my portfolio; happy to walk through any of them.

Thanks for the read,
Derek Lankeaux
