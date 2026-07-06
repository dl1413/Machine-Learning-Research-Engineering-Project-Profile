# Cover Letter — S&P Global
**Role:** Data Scientist — NLP, LLM and GenAI
**Location:** New York, NY
**Source:** efinancialcareers.com/jobs-USA-NY-New_York-Data_Scientist_-_NLP_LLM_and_GenAI.id20854798
**Anchor Projects:** LLM Ensemble Bias Detection · AI Safety Red-Team Evaluation

---

Dear S&P Global Hiring Team,

I'm applying for the Data Scientist — NLP, LLM and GenAI role. My work sits exactly at the intersection you describe: applying multi-model LLM ensembles to problems where **auditability and statistical rigor** matter as much as raw model performance — a fit for evaluating financial content, credit narratives, and analyst commentary at S&P's scale.

**LLM ensemble evaluation, production-grade.** In my LLM Ensemble Textbook Bias Detection project I processed **67,500 ratings across 4,500 passages (2.5M tokens)** using a GPT-4o / Claude-3.5 / Llama-3.2 ensemble, with **92% pairwise correlation** and **Krippendorff's α = 0.84** — the same "does this rating hold up across judges?" question S&P faces when signals feed downstream products. I built the underlying pipeline with circuit breakers, exponential backoff, and MLflow tracking, and quantified publisher-level bias with a **Bayesian hierarchical model (PyMC, R-hat < 1.01, χ² = 42.73, p < 0.001)** — 3 of 5 publishers showed credibly non-zero bias at the 95% HDI level.

**Cost-to-quality tradeoffs on real annotation budgets.** My AI Safety Red-Team framework hit **340× cost reduction** ($0.018/sample vs. $6.12 human) at **α = 0.81** across 12,500 pairs and 6 harm categories, with a Stacking Classifier reaching **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**. The reusable pattern — LLM-as-judge front-end, ML aggregator back-end, human review only on the uncertain tail — is directly applicable to financial-content moderation, sentiment on earnings transcripts, and NLP-driven risk taxonomies.

**What I bring day one:** SQL + Python fluency, offline eval design, prompt iteration, HDI/calibration reporting, and stakeholder-ready model cards aligned to IEEE 2830-2025 — increasingly the default in regulated data work.

I'd welcome the chance to discuss the role.

Best regards,
**Derek Lankeaux, MS**
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
