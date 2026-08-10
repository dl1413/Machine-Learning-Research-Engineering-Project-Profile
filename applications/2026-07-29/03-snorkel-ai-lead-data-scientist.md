# Snorkel AI — Lead Data Scientist

**Location:** Remote (US)
**Apply URL:** https://job-boards.greenhouse.io/snorkelai/jobs/6104260004
**Team charter (per posting):** Data-centric AI — programmatic labeling, LLM-annotation-as-supervision, evaluation, and applied deployments on enterprise data.

*Seniority note: the posting reads "Lead," which typically implies 6+ YOE. Domain fit is exceptional — the letter leans on portfolio depth and is explicit about the seniority delta. If a lower-level Snorkel DS req is open the day you apply, prefer that one and reuse this letter.*

---

## Cover Letter

Dear Snorkel Hiring Team,

I'm applying to the Lead Data Scientist role. I'm early in my career — finishing an MS in Applied Statistics at RIT — so I'll be upfront that the "Lead" title is a stretch on paper. I'm writing anyway because the work I've done on my own is essentially the Snorkel playbook applied to LLM evaluation, and I'd rather flag the seniority gap and let you decide than let the title filter me out.

The most Snorkel-shaped thing I've built is an **AI Safety Red-Team framework**: a **GPT-4o + Claude-3.5 + Llama-3.2 ensemble annotates** each response and a stacked ML classifier makes the final call. On **12,500 samples and 6 harm categories** it hits **96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923)** with **Krippendorff's α = 0.81**, at **$0.018/sample** — a **340× cost reduction** vs. human labeling — and **850 samples/hour**. That's programmatic supervision, weak-signal aggregation, and evaluation as one loop — the same shape as a Snorkel workflow, just applied to harm signals instead of enterprise entities.

A second project sits closer to Snorkel's Bayesian-eval story: a **multi-LLM bias-detection system** on **4,500 textbook passages (67,500 ratings, 2.5M tokens)** where a **PyMC hierarchical model with partial pooling (R-hat < 1.01)** returns **95% HDI credible intervals per publisher** — the uncertainty-aware layer on top of noisy LLM labels that a data-centric eval platform depends on. Pairwise LLM agreement was **92%**, and I used circuit breakers, exponential backoff, and MLflow tracking to run it at the API scale you need to actually generate 2.5M tokens.

I also carry the classical-ML muscle: my **Breast Cancer classification** stack — 99.12% accuracy, 100% precision, Platt-calibrated ECE 0.0089, SHAP for clinical transparency — shows I've worked the calibration and explainability side of the pipeline, not just the LLM-annotation front end.

I'd welcome a conversation, and I'm happy to interview against a more junior req if that's a better fit for the team's current headcount.

Regards,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux · GitHub: https://github.com/dl1413

---

## Role-tailored resume bullets

- Programmatic-supervision pipeline on **12,500 LLM outputs**: ensemble annotation + stacked-ML classifier at **96.8% accuracy** and **α = 0.81**, running **340×** cheaper than human labeling.
- Uncertainty-aware LLM-eval layer on **67.5K ratings** using PyMC hierarchical partial pooling (**R-hat < 1.01, 95% HDI**) — the credible-interval discipline data-centric platforms need on top of noisy labels.
