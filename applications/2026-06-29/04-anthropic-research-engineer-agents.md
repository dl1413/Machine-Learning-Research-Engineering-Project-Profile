# Cover Letter — Anthropic, Research Engineer, Agents

**Role:** Research Engineer, Agents
**Location:** NYC / SF (office-first)
**Apply:** https://job-boards.greenhouse.io/anthropic/jobs/4017544008
**Lead projects:** P1 (AI Safety Red-Team Evaluation), P2 (LLM Ensemble Bias Detection)

> Note: Stretch application. The JD calls out "scaled model evaluation framework driven by model-based evaluation techniques," which is the literal shape of P1. Worth a shot even at the experience asymmetry.

---

Dear Anthropic Hiring Team,

I'm applying for the Research Engineer, Agents role. The JD's phrase "scaled model evaluation framework driven by model-based evaluation techniques" is exactly what I built and shipped this year, twice.

**P1 — AI Safety Red-Team Evaluation.** A dual-stage framework where an LLM ensemble (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) performs first-pass annotation and a Stacking Classifier does triage. Evaluated 12,500 AI response pairs across six harm categories (dangerous info, hate, deception, privacy, illegal activity, self-harm) at 96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923). The annotator ensemble itself held Krippendorff's α = 0.81 — the inter-rater-reliability number that determines whether model-based eval can replace human eval at all. Throughput 850 samples/hour at $0.018/sample, a 340× cost reduction over human annotation.

**P2 — LLM Ensemble Bias Detection.** Same scaffold, different domain: 67,500 ratings × 4,500 textbook passages × 5 publishers, with a Bayesian hierarchical model in PyMC (partial pooling, R-hat < 1.01, 95% HDI) on top to convert noisy LLM judgments into publisher-level credible intervals. Friedman χ² = 42.73, p < 0.001. The point of this project was proving the P1 framework generalizes — that the multi-rater + Bayesian pooling pattern isn't specific to harm taxonomy.

What I think is relevant to the Agents team specifically:

- **Model-based eval as a primitive, not a one-off.** Both projects treat LLM-as-judge as an instrument that needs calibration (α, R-hat, pairwise correlation) before its outputs are used as ground truth for downstream classification.
- **8-category MITRE ATLAS-aligned attack taxonomy.** Multi-turn escalation identified as the dominant failure mode at 31.8% — the kind of adversarial structure agents have to be robust against.
- **Production plumbing.** Circuit breakers, exponential backoff, MLflow experiment tracking, 80K+ API calls / 2.5M tokens processed without manual intervention.

I'm finishing an MS in Applied Statistics at RIT in 2026. I'm aware the experience bar at Anthropic is high; I'm submitting because the project shape genuinely matches the JD and would rather have you make the call. Reports (IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act aligned) available as PDFs.

Best,
Derek Lankeaux, MS (Applied Statistics, RIT — 2026)
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
