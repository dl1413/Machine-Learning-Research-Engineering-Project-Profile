# Ramp — Data Scientist (Product)

**Posting:** https://jobs.ashbyhq.com/ramp/e577622f-6657-4e53-8941-b3a774b04448
**Location:** New York, NY
**Date drafted:** 2026-07-08

---

Dear Ramp Data Science team,

I'm applying for the Data Scientist role on the Product team. Ramp's operating loop — ship a feature that touches $200B+ of annualized spend across 70,000+ companies, then defend the result with an experiment that stakeholders trust — is exactly the workflow my MS in Applied Statistics prepared me for, and it's the through-line across my three most recent projects.

**What I bring to Product DS:**

- **Experimentation and causal rigor.** Frequentist and Bayesian hypothesis testing, multiple-testing correction (Bonferroni, FDR, Holm-Sidak), effect sizes (Cohen's d, η²), power/sample-size calculations, and quasi-experimental designs when randomization isn't possible. In my LLM bias study I used Friedman χ² = 42.73 (p < 0.001), Bayesian hierarchical partial pooling (PyMC, R-hat < 1.01, 95% HDI), and bootstrap CIs — the same toolkit I'd bring to a Ramp A/B or holdout readout to keep the decision robust to peeking and multiple comparisons.
- **End-to-end ML for product features.** My clinical breast-cancer classifier reached **99.12% accuracy with 100% precision and Platt-calibrated probabilities (ECE 0.0089)** across an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) with VIF, SMOTE, and RFE in the pipeline. That's the same pattern I'd use for a fraud/anomaly or spend-categorization model — calibrated probabilities and context-tuned thresholds so business partners can pick their operating point.
- **LLM-powered product features, cost-aware.** My AI Safety Red-Team pipeline hit **96.8% accuracy at $0.018/sample vs $6.12 for human annotation (340× cost reduction)** — directly applicable to any Ramp workflow where an LLM sits between a user and a decision (receipt parsing, policy checks, categorization) and you need offline eval + monitoring that scale with usage.
- **Communication.** 3 publication-grade reports with model cards, calibration plots, and SHAP explanations written for non-technical reviewers — the readout that closes a product decision, not just the notebook that produced it.

Portfolio and code: https://dl1413.github.io/LLM-Portfolio/ • https://github.com/dl1413. Resume attached.

Best,
Derek Lankeaux
