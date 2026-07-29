# Capital One — Applied Researcher I (AI Foundations, LLM Core & Agentic AI)

**Location:** New York, NY
**Apply URL:** https://www.capitalonecareers.com/en/job/new-york/applied-researcher-i-ai-foundations-llm-core-and-agentic-ai/1732/93698173424
**Team charter (per posting):** AI Foundations — high-impact applied research on the latest LLM/agentic AI developments, working with PyTorch on AWS Ultraclusters, moving research into next-generation customer experiences.

---

## Cover Letter

Dear AI Foundations Hiring Team,

I'm writing to apply for the Applied Researcher I role on the LLM Core & Agentic AI team. My background is Applied Statistics rather than a pure ML PhD, but the way I've been working — Bayesian evaluation of LLM systems, ensemble-as-judge pipelines, and reproducible reporting for regulated contexts — is close to the "research life cycle" this team owns, and I'd like to bring that discipline to Capital One's agentic AI work.

The project I'd lean on hardest is an **LLM Ensemble Bias Detection framework**: 4,500 textbook passages, **67,500 ratings, 2.5M tokens** processed through GPT-4o + Claude-3.5 + Llama-3.2, then modeled with a **PyMC hierarchical model with partial pooling**. Convergence checked out (**R-hat < 1.01**), pairwise LLM correlation ran at **92%**, and the posterior surfaced **statistically significant bias in 3/5 publishers (Friedman χ² = 42.73, p < 0.001)** with **95% HDI** on every claim. For an agentic system, that's the shape of the evaluation you need before you trust an LLM's judgment inside a loop — a credible interval, not just an average.

Complementing that is an **AI Safety Red-Team** framework running dual-stage ensemble annotation + a stacked ML classifier at **96.8% accuracy** across **12,500 pairs and 6 harm categories**, at **$0.018/sample (340× cheaper than human labels)** and **850 samples/hour** with SHAP explanations. It's the LLM-as-judge → learned-classifier pattern that lets a research org iterate on model changes at a cost per experiment low enough to actually run the experiments.

I'm finishing my MS in Applied Statistics at RIT (Bayesian Inference & MCMC, Deep Learning, Causal Inference), which gave me the toolkit — power analysis, multiple-testing correction, calibration (Platt / isotonic; see my **Breast Cancer classifier's ECE 0.0089**), Bayesian hierarchical models — that Applied Researcher work depends on when a claim has to survive real scrutiny.

I'd welcome a conversation about the team's current LLM-eval and agent-evaluation priorities.

Regards,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux · GitHub: https://github.com/dl1413

---

## Role-tailored resume bullets

- Modeled **67.5K LLM bias ratings** with a PyMC hierarchical model (R-hat < 1.01, 95% HDI) that surfaced 3/5 credible publisher-level effects — the credible-interval discipline agentic-AI evaluation requires.
- Built an LLM-ensemble-as-judge → stacked-ML pipeline at **96.8% accuracy** and **$0.018/sample**, giving a research team an eval budget low enough to actually run the experiments.
