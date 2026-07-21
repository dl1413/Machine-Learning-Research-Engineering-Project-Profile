# Cover Letter — Etsy, Data Scientist (Product Analytics)

**Role:** Data Scientist, Product Analytics
**Company:** Etsy
**Location:** Brooklyn, NY / Hybrid (NYC area)
**Source:** https://startup.jobs/locations/new-york/data-scientist
**Date drafted:** 2026-07-21

---

Dear Etsy Data Science Hiring Team,

I'm applying for the Data Scientist, Product Analytics role. Two-sided marketplaces run on funnel and cohort work that lives or dies on how carefully the effects are measured — that's what my Applied Statistics MS (RIT, 2026) has prepared me for, and it's what three shipped projects demonstrate.

**Measurement in noisy, human-labeled settings.** In *LLM Ensemble Textbook Bias Detection* I processed 67,500 ratings across 4,500 passages, discovering that only 3/5 publishers had credibly non-zero effects after Bayesian pooling — the other two were consistent with noise despite frequentist significance in the raw data. That's the same shape as buyer-behavior data on Etsy: category effects that look real until you pool across sellers.

**Causal and quasi-experimental design.** I've written Friedman χ² tests, Bonferroni/FDR corrections, bootstrap CIs, and power-analysis writeups; the *AI Safety Red-Team* project ran a defense-vs-attack quasi-experiment showing harm rate drop from 21.8% to 4.8% under a dual-filter policy (78% reduction) with the appropriate CIs. That's the kind of "was it the intervention or the composition?" question I'd bring to Etsy's product tests.

**Threshold policy on live decisions.** *Breast Cancer ML Classification* (99.12% accuracy, 100% precision) used Platt-calibrated scores and context-adaptive thresholds — one threshold for screening, another for confirmation. Product classifiers on Etsy (fraud, recs, search quality) work the same way: the model isn't the product; the decision policy on top of it is.

**Communication.** I've published three technical reports written for both technical reviewers and business stakeholders — model cards, calibration plots, SHAP-based explanations — the artifact your PMs and marketing partners actually read.

I'd like the chance to bring this toolkit to Etsy's Product Analytics team.

Sincerely,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux | GitHub: https://github.com/dl1413

---

## Talking points for phone screen

- **Why Etsy:** two-sided marketplace where seller-side and buyer-side effects both matter — Bayesian partial pooling is the right hammer.
- **Product analytics story:** every project reports HDI/CIs, not just a point estimate.
- **Cross-functional writing:** three publication-grade reports for mixed audiences.
