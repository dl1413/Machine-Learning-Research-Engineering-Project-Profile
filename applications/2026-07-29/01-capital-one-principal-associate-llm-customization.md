# Capital One — Principal Associate, Data Scientist (LLM Customization Team)

**Location:** New York, NY
**Apply URL:** https://www.capitalonecareers.com/job/new-york/principal-associate-data-scientist-llm-customization-team/1732/92083762528
**Team charter (per posting):** AI Foundations LLM Customization — adapting and fine-tuning frontier LLMs for Capital One's business-specific applications, working across the research life cycle from research through production.

---

## Cover Letter

Dear Capital One AI Foundations Hiring Team,

I'm applying to the Principal Associate, Data Scientist role on the LLM Customization team. The team's charter — adapting frontier LLMs into production-ready, business-specific systems — maps directly to the work I've been doing on my own for the last year, and I'd like to bring it into a regulated, real-stakes environment.

My most relevant project is an **AI Safety Red-Team Evaluation framework** that runs a dual-stage pipeline: a GPT-4o + Claude-3.5 + Llama-3.2 ensemble annotates responses, then a stacked ML classifier makes the final call. On 12,500 evaluations across six harm categories, it reached **96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923)** while cutting per-sample cost **340×** vs. human annotation (**$0.018 vs. $6.12**) and holding **Krippendorff's α = 0.81** — the LLM-ensemble-as-judge pattern that a customization team needs when you want to iterate on prompts and fine-tunes without paying a human-eval bill for every change. The pipeline runs at 850 samples/hour with SHAP explanations and MLflow tracking, so the audit trail exists before compliance asks for it.

Behind that sits an **LLM Ensemble Bias Detection** system on 4,500 textbook passages (67,500 ratings, 2.5M tokens) where I used a **PyMC hierarchical model with partial pooling (R-hat < 1.01, 95% HDI)** to surface publisher-level bias that a naive average missed — the same posterior-uncertainty story that matters when a fine-tune decision has to survive a governance review. It's a reasonable proxy for the kind of statistical rigor Capital One would want on a customer-facing LLM change.

I'm finishing my MS in Applied Statistics at RIT (Bayesian methods, causal inference, deep learning). My **Breast Cancer ML** project — 99.12% accuracy with a Platt-calibrated ECE of 0.0089 — shows I can also carry the classical-ML and calibration side of a customization stack, not just the LLM half. I would welcome a conversation about the team's current evaluation and fine-tuning priorities.

Regards,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux · GitHub: https://github.com/dl1413

---

## Role-tailored resume bullets (swap into the "Skills highlights" line if you want a one-page variant)

- Built a production LLM-as-judge ensemble that replaced $6.12 human annotation with a $0.018/sample pipeline at **α = 0.81** — the pattern LLM-customization teams need to iterate on prompts and fine-tunes without a per-experiment eval bill.
- Delivered publisher-level bias findings on **67.5K LLM ratings** using a **PyMC hierarchical model (R-hat < 1.01, 95% HDI)** — statistical rigor a governance review can defend for a customer-facing LLM change.
