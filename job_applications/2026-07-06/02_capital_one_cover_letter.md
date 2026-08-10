# Cover Letter — Capital One
**Role:** Principal Associate, Data Scientist — LLM Customization Team (AI Foundations)
**Location:** New York, NY
**Source:** capitalonecareers.com/job/new-york/principal-associate-data-scientist-llm-customization-team/1732/92083762528
**Anchor Projects:** AI Safety Red-Team Evaluation · LLM Ensemble Bias Detection

---

Dear Capital One AI Foundations Team,

The AI Foundations LLM Customization posting emphasizes end-to-end ownership — research to production system — which is exactly the shape of the last three projects I've shipped. I'd like to bring that pattern to Capital One's customer-facing and internal LLM surfaces, where **calibration, drift, and adversarial robustness** determine whether a model can safely ship at bank scale.

**Red-team evaluation you can actually deploy.** My AI Safety Red-Team framework evaluated **12,500 AI response pairs** across a GPT-4o / Claude-3.5 / Llama-3.2 ensemble, hitting **96.8% accuracy (Stacking, ROC-AUC 0.9923)** at **340× cost reduction** vs. human annotation. I engineered **47 linguistic/semantic/structural features**, a MITRE ATLAS-aligned 8-category adversarial taxonomy, and a dual-filter defense that reduced observed harm rate from **21.8% → 4.8% (78% reduction)** — the same shape of guardrail evaluation Capital One will need for retrieval- and tool-augmented finance assistants.

**Statistical rigor over vibes-based eval.** My LLM Bias Detection work produced **67,500 ratings, α = 0.84**, and a PyMC Bayesian hierarchical model with **R-hat < 1.01** and 95% HDI credible intervals — the kind of uncertainty quantification that lets a regulated org distinguish a real regression from noise. I paired that with **12.3% high-uncertainty flagging** for human review, which is the pattern most bank-grade HITL loops need.

**Production hygiene.** Circuit breakers, exponential backoff across **80K+ API calls / 2.5M tokens**, MLflow experiment tracking, FastAPI serving <100ms p95, SHAP + model cards aligned to IEEE 2830-2025 and ISO/IEC 23894:2025 — the compliance-friendly artifacts a bank's model risk team expects.

I'd love to discuss how this maps to the LLM Customization team's roadmap.

Best regards,
**Derek Lankeaux, MS**
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
