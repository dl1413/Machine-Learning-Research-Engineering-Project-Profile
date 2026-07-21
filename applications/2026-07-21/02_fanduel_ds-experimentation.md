# Cover Letter — FanDuel, Data Scientist (Experimentation)

**Role:** Data Scientist, Experimentation
**Company:** FanDuel
**Location:** New York, NY (hybrid)
**Source:** https://www.ziprecruiter.com/Jobs/Data-Scientist-Experimentation
**Date drafted:** 2026-07-21

---

Dear FanDuel Data Science Hiring Team,

I'm applying for the Data Scientist, Experimentation role. FanDuel's product tempo — high-throughput A/B testing where a small effect on wager frequency compounds into real revenue — is the environment where the Bayesian and multi-arm methods I trained on pay off. My MS in Applied Statistics (RIT, 2026) sits on three shipped projects.

**Experimentation and multiple-testing discipline.** In *LLM Ensemble Textbook Bias Detection* I designed a comparison across 5 publishers and 3 topics: Friedman χ² = 42.73 (p < 0.001), Bonferroni/FDR correction, 95% HDI on every effect, and 12.3% of passages flagged as high-uncertainty for expert review rather than pushed through. That's the same review that separates a real product win from a noisy readout.

**Bayesian experimentation.** The same project used PyMC hierarchical models with partial pooling and MCMC diagnostics (R-hat < 1.01, ESS > 400 per parameter). Partial pooling is exactly how I'd stabilize per-cohort estimates for a variant that's only shown to a slice of users, and HDI intervals give product partners the calibrated "how sure are we?" they actually need.

**Model calibration and threshold policies.** *Breast Cancer ML Classification* (99.12% accuracy, ROC-AUC 0.9987) used Platt scaling to drop ECE 71.5% (0.0312 → 0.0089) and context-adaptive thresholds — the same trick I'd use to convert a lift model into a decision policy that respects your risk and compliance constraints.

**GenAI as an additional lever.** *AI Safety Red-Team Evaluation* (96.8% accuracy across 12,500 pairs, 340× cost reduction) shows I can operate LLM-as-judge pipelines when a human-eval budget is the bottleneck — useful for FanDuel's trust/safety and content moderation surfaces.

Every project ships with SQL, MLflow tracking, versioned artifacts, and a stakeholder-ready readout. I'd like to bring that to FanDuel's experimentation platform.

Sincerely,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux | GitHub: https://github.com/dl1413

---

## Talking points for phone screen

- **Why FanDuel:** experimentation at consumer-product velocity, with real revenue on the line.
- **Bayesian A/B:** partial-pooling and HDI is my default, not an exotic option.
- **Multiple-testing:** Bonferroni/FDR is baked into my reports — I've defended it in writing.
