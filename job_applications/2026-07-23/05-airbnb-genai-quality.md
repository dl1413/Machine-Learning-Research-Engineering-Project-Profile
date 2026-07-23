# Airbnb — Data Scientist, GenAI Quality & Evaluation

**Location:** New York, NY · **Apply:** https://careers.airbnb.com/ (search
"GenAI" / "LLM" DS roles) · **Fit:** ★★★★☆

---

Dear Airbnb DS hiring team,

I'm applying for a Data Scientist role focused on GenAI quality and evaluation.
Airbnb's LLM surfaces (search, guest/host messaging, review summarization, agentic
support) all live or die by the same thing: an offline-evaluation and A/B pipeline
that can defend a launch decision at the review meeting.

I've built exactly that pattern end-to-end. My **AI Safety Red-Team Evaluation** and
**LLM Ensemble Bias Detection** projects are two sides of a production LLM-evaluation
harness: three frontier models (GPT-4o, Claude-3.5, Llama-3.2) running as an
ensemble, LLM-as-judge with reliability quantified (**Krippendorff's α = 0.81 and
0.84**), a downstream Stacking Classifier at **96.8% accuracy** for the safety side,
and a **PyMC hierarchical model (R-hat < 1.01, 95% HDI, Friedman χ² = 42.73,
p < 0.001)** with bootstrap uncertainty for the fairness side. 80K+ API calls,
2.5M tokens, circuit breakers, exponential backoff, and MLflow tracking — repeatable
enough to gate a launch, cheap enough to run continuously ($0.018/sample, 850/hour).

The DS surface Airbnb interviews on — A/B design, power analysis, multiple-testing
correction (Bonferroni/FDR), effect sizes, Bayesian hierarchies for
market/host/segment effects, SQL at scale, and clear readouts — is the toolkit I
practiced in all three projects (the third being a clinical classifier at 99.12%
accuracy with Platt calibration reducing ECE by 71.5%). Publication-grade model
cards and calibration plots are in the repo, written for both technical reviewers
and non-technical stakeholders.

Would love to dig into how offline eval + online experimentation intersect for one
of the GenAI surfaces in a first conversation.

Derek Lankeaux · MS Applied Statistics, RIT (2026) ·
[LinkedIn](https://linkedin.com/in/derek-lankeaux) ·
[GitHub](https://github.com/dl1413)
