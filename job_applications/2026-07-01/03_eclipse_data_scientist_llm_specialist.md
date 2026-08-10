# Cover Letter — Eclipse, Data Scientist (AI Data & LLM Specialist)

**Job link:** https://job-boards.greenhouse.io/eclipse/jobs/4981191008

---

Dear Eclipse Hiring Team,

I'm applying for the Data Scientist (AI Data & LLM Specialist) role. The posting frames the problem well: raw data collections don't teach models much until someone builds the labeling, processing, and evaluation scaffolding around them. That has been the throughline of my last year of work.

**Turning raw collections into audit-grade labeled data.** In my LLM Ensemble Textbook Bias Detection project I processed 4,500 passages (2.5M tokens) through a multi-LLM annotation pipeline — GPT-4o, Claude-3.5, Llama-3.2 — producing 67,500 ratings with Krippendorff's α = 0.84 and 92% pairwise correlation. A PyMC hierarchical model (partial pooling, R-hat < 1.01) then produced publisher-level credible intervals and flagged 12.3% of passages for expert review via bootstrap CIs. That routing loop — automated labeling with statistically defensible escalation — is the pattern I'd bring to Eclipse.

**Evaluation-first LLM engineering.** My AI Safety Red-Team framework built a benchmark of 12,500 model responses across six harm categories, achieving 96.8% accuracy and 340× cost reduction ($0.018/sample vs. $6.12 for human annotation) while preserving reliability (Krippendorff's α = 0.81). The pipeline includes 47 engineered features (linguistic, semantic, structural), a MITRE ATLAS-aligned taxonomy, and SHAP explanations — the shape of a dataset that's ready to train, fine-tune, and be audited against.

**Production-grade tooling.** Both projects run with circuit breakers, exponential backoff, rate limiting, and MLflow experiment tracking across 80K+ API calls. My clinical breast-cancer classifier (99.12% accuracy, 100% precision, ECE 0.0089) is deployed behind FastAPI at < 100ms p95. I'm comfortable turning a labeling notebook into a monitored service.

I'm completing an MS in Applied Statistics at RIT (Bayesian and experimental design focus) in 2026 and looking for teams where dataset quality is treated as the leverage point for model quality. Portfolio, reports, and code at github.com/dl1413.

Best regards,
Derek Lankeaux
dlankeaux12@gmail.com | linkedin.com/in/derek-lankeaux
