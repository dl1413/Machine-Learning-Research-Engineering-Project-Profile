# Ramp — Senior Data Scientist, Growth

**Location:** New York, NY (hybrid) / US Remote
**Apply:** https://www.remoterocketship.com/us/company/ramp/jobs/senior-data-scientist-growth-new-york-city-hybrid/
**Lead project:** LLM Ensemble Textbook Bias Detection (Bayesian hierarchical + experimental rigor)
**Supporting projects:** Breast Cancer Classification (calibrated probabilities for growth-policy thresholds), AI Safety Red-Team (LLM-assisted analytics at scale)

---

Dear Ramp Growth team,

Ramp's Growth Data Science bar reads like the checklist I've spent the last year building against: canonical experimentation standards, causal inference, Bayesian methods, and turning noisy marketing signal into decisions finance and engineering can defend. That's exactly the discipline I've been sharpening as an MS in Applied Statistics — and I've shipped three end-to-end research projects that show it in production form.

The closest analog to your Growth problem is a Bayesian hierarchical evaluation system I built and published in April 2026. It processes 67,500 bias ratings across 4,500 passages using a PyMC partial-pooling model that converges cleanly (R-hat < 1.01), quantifies uncertainty with 95% HDIs, and produces publisher-level credible bias estimates. The pipeline detected statistically significant differences (Friedman χ² = 42.73, p < 0.001) in 3 of 5 publishers under Bonferroni/FDR correction — the same multiple-testing discipline that keeps growth experiments honest when you're running many arms in parallel. Krippendorff's α held at 0.84 across GPT-4o, Claude-3.5, and Llama-3.2 raters, with circuit breakers and MLflow lineage on the ingestion side so results are auditable end to end.

For growth-policy decisions where you're setting spend or targeting thresholds, calibration matters as much as ranking. On my breast cancer classification work I ran an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) that hit 99.12% accuracy and ROC-AUC 0.9987, then reduced expected calibration error 71.5% (0.0312 → 0.0089) with Platt scaling and tuned context-adaptive thresholds — the same recipe that lets a growth team say "at this predicted LTV, spend $X" rather than "the model ranked it high." I'd bring that same "quantify, calibrate, defend" mindset to Ramp's marketing-investment frameworks, and I'm comfortable owning it end-to-end in SQL, Python (Pandas/Polars), PyMC, and MLflow.

Happy to walk through any of the three published reports (all on GitHub) in a first conversation.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
