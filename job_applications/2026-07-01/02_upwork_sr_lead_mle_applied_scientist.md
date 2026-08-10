# Cover Letter — Upwork, Sr Lead ML Engineer / Applied Scientist (Agent Evaluation)

**Job link:** https://job-boards.greenhouse.io/upwork/jobs/7565868003

---

Dear Upwork AI Hiring Team,

I'm applying for the Sr Lead ML Engineer / Applied Scientist role focused on agent evaluation. Building trustworthy evaluation harnesses for LLM systems is what I've spent the last twelve months on, and the description reads almost line-for-line like the work I've been publishing.

**Reproducible evaluation frameworks.** My AI Safety Red-Team framework runs 12,500 responses through a dual-stage pipeline (GPT-4o + Claude-3.5 + Llama-3.2 annotation → Stacking Classifier over 47 engineered features) and reports 96.8% overall accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923, and a Krippendorff's α of 0.81 across the LLM judges. Every stage has an audit trail (MLflow), a defense-effectiveness analysis (dual-filter reduces harm rate 21.8% → 4.8%), and SHAP-based explanations. The important artifact isn't the number — it's a benchmark that can be rerun, versioned, and defended.

**Statistical experimentation at production scale.** For LLM Ensemble Bias Detection I designed the offline benchmark end-to-end: 67,500 ratings across 4,500 passages and 2.5M tokens, with a PyMC hierarchical model (partial pooling, R-hat < 1.01), Friedman χ² = 42.73 (p < 0.001), Bonferroni/FDR correction, bootstrap CIs, and passage-level uncertainty flags that route 12.3% of items to human review. That "who should look at this next" loop is exactly what's needed to grade agents doing real-world tasks.

**Production reliability, not just notebooks.** Both projects run behind circuit breakers, exponential backoff, and rate limiting; the pipelines processed 80K+ API calls without an incident. My third project — a breast-cancer classifier at 99.12% accuracy / 100% precision — sits behind FastAPI at < 100ms p95 with an MLflow registry. I understand what an evaluation system has to look like when it lives next to production, not next to a paper.

I'm finishing an MS in Applied Statistics at RIT in 2026. Upwork's angle — defining success for agents doing consequential real-world work — is the kind of problem I'd like to spend the next several years on. Portfolio and technical reports: github.com/dl1413.

Best regards,
Derek Lankeaux
dlankeaux12@gmail.com | linkedin.com/in/derek-lankeaux
