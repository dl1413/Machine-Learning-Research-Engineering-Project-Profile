# Uber — Senior Applied Scientist, AI Red Teaming & Model Risk

**Location:** New York, NY
**Posting:** https://www.uber.com/global/en/careers/list/ (search "AI Red Teaming Model Risk NYC")
**Lead project:** AI Safety Red-Team Evaluation + LLM Ensemble Bias Detection

---

## Cover Letter

Hi Uber Applied Science team,

The Senior Applied Scientist, AI Red Teaming & Model Risk posting lines up with the exact intersection I've been building toward: adversarial evaluation of generative systems, model-risk quantification, and the statistical machinery to defend a risk claim in front of legal, product, and safety stakeholders.

**Direct red-team experience.** My **AI Safety Red-Team Evaluation** framework (April 2026) shipped a dual-stage LLM ensemble + ML classifier over **12,500 adversarial response pairs**, hitting **96.8% accuracy** and processing **850 samples/hour** with SHAP explainability and audit trails. I built the 8-category MITRE ATLAS-aligned attack taxonomy, isolated multi-turn escalation as the top risk (31.8%), and measured that a dual-filter defense cut harm from **21.8% → 4.8%**. Cost per evaluation dropped **340×** vs. human labeling with reliability held at Krippendorff's α = 0.81.

**Model-risk quantification, not just detection.** In my **LLM Ensemble Textbook Bias Detection** work, I built a PyMC hierarchical model with partial pooling that turned **67,500 ratings across 4,500 passages** into **publisher-level credible bias intervals (95% HDI)**, with **MCMC R-hat < 1.01** and Friedman χ² = 42.73 (p < 0.001). That's the same statistical shape a model-risk function needs: bounded credible intervals on behavioral risk per model version, not point estimates.

**Production-grade engineering.** 80K+ API calls / 2.5M tokens processed with circuit breakers, exponential backoff, and MLflow tracking. FastAPI serving under 100 ms p95 on the Breast Cancer classifier (99.12% accuracy, ECE 0.0089 after Platt calibration) — I ship, calibrate, and monitor, not just prototype.

I'm an Applied Statistics MS at RIT (expected 2026), authorized to work in the US, and all three technical reports are IEEE 2830-2025 / ISO/IEC 23894 / EU AI Act-aligned.

Thank you for the read,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Attach

- Resume PDF
- `AI_Safety_RedTeam_Evaluation_Publication.pdf`
- `LLM_Bias_Detection_Publication.pdf`
