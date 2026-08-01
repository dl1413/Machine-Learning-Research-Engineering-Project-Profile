# Peloton — Data Scientist, Product Analytics

**Location:** New York, NY
**Apply:** https://www.builtinnyc.com/job/data-scientist-product-analytics/3697366
**Lead project:** LLM Ensemble Textbook Bias Detection (Bayesian hierarchical + experimental rigor)
**Supporting projects:** Breast Cancer Classification (calibrated retention/churn thresholds), AI Safety Red-Team (LLM-assisted product analytics at scale)

---

Dear Peloton Product Analytics team,

Habit formation and engagement analytics are Bayesian-inference problems dressed up as A/B tests — user variance is high, sample sizes per cohort are small, and you need to say something honest about "does this feature actually move Week-4 retention?" without over-claiming from a single experiment. That's the exact modeling posture I've been trained in, and it's why I'm applying for the Data Scientist, Product Analytics role.

The most transferable piece of my portfolio is a multi-LLM evaluation framework I built and published in April 2026. It processes **67,500 ratings across 4,500 passages (2.5M tokens)** using a PyMC hierarchical model with partial pooling — the same tool product analytics teams use when they want to share statistical strength across cohorts without pretending they're identical. MCMC convergence was clean (R-hat < 1.01), and the framework produced credible-interval-based conclusions (95% HDIs) rather than fragile point estimates. On the hypothesis-testing side I ran a Friedman non-parametric test (χ² = 42.73, p < 0.001) with Bonferroni/FDR correction across five sources — exactly the multiple-testing discipline you need when you're evaluating 5–10 feature variants in the same quarter.

For the "will this retention model actually make good decisions?" question, my breast cancer classification work is the more direct analog. I trained an 8-algorithm ensemble reaching 99.12% accuracy and ROC-AUC 0.9987, then invested heavily in calibration and threshold tuning: Platt scaling reduced expected calibration error 71.5% (0.0312 → 0.0089), and context-adaptive thresholds mapped model outputs to different decision policies. Swap "clinical screening" for "retention intervention" and it's the same problem — you don't want to trigger a coupon or a push at the wrong probability.

I've shipped these projects end-to-end in Python (Pandas/Polars), SQL, PyMC, scikit-learn, MLflow, and FastAPI, and documented each as a publication-grade technical report on GitHub. MS in Applied Statistics (RIT, 2026); hybrid-friendly for the NYC office. Would love to talk about how Product Analytics could use this toolkit.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
