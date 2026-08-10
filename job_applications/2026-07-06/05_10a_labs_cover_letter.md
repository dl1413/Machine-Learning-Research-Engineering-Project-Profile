# Cover Letter — 10a Labs
**Role:** AI Red Teamer (Entry Level / High-Impact)
**Location:** Remote (United States)
**Source:** job-boards.greenhouse.io/10alabs/jobs/4002004009
**Anchor Projects:** AI Safety Red-Team Evaluation (direct 1:1 match) · LLM Ensemble Bias Detection

---

Dear 10a Labs Team,

The AI Red Teamer posting is the closest 1:1 match to the framework I've spent the last year building. I want to bring it to a team whose day-to-day is **adversarial testing, abuse detection, and multilingual threat intelligence at scale.**

**Adversarial taxonomy + measured defense uplift.** My AI Safety Red-Team Evaluation Framework covers exactly what 10a Labs ships:

- **12,500 AI response pairs** evaluated across **6 harm categories** (dangerous info, hate, deception, privacy, illegal activity, self-harm)
- **8-category MITRE ATLAS-aligned adversarial taxonomy**; identified **multi-turn escalation as the highest-risk vector at 31.8%**
- **Dual-filter defense** that reduced observed harm rate from **21.8% → 4.8% (78% reduction)** — measured, not asserted
- **Krippendorff's α = 0.81** ensemble reliability across GPT-4o, Claude-3.5, and Llama-3.2
- **96.8% accuracy Stacking Classifier** (97.2% precision, 96.1% recall, ROC-AUC 0.9923) on **47 engineered linguistic/semantic/structural features**
- **340× cost reduction** ($0.018/sample) — the unit economics that let a small red-team scale coverage

**Vulnerability writing you can hand to eng.** Each finding is scored with a **Bayesian hierarchical model (PyMC, 95% HDI)** so severity ordering isn't just gut feel. I've written **3 publication-grade technical reports** with model cards, SHAP-based explanations, and IEEE 2830-2025 / EU AI Act-aligned artifacts — evidence I can produce both short vulnerability write-ups and long-form analyses.

**Engineering.** Python 3.12+, MLflow, FastAPI, Docker; production API integration with circuit breakers and exponential backoff across **80K+ calls / 2.5M tokens**. Comfortable working async and independently in a remote setup.

I'd love to talk about how this framework can plug directly into 10a Labs' red-team engagements.

Best regards,
**Derek Lankeaux, MS**
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
