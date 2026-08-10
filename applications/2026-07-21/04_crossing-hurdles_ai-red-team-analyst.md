# Cover Letter — Crossing Hurdles, AI Red Team Analyst (LLM Safety)

**Role:** AI Red Team Analyst — LLM Safety / Adversarial Testing
**Company:** Crossing Hurdles
**Location:** Remote (US)
**Rate (posted):** $55/hr
**Source:** https://www.linkedin.com/jobs/view/ai-red-team-analyst-llm-safety-adversarial-testing-$55-55-hr-remote-at-crossing-hurdles-4372563042
**Date drafted:** 2026-07-21

---

Dear Crossing Hurdles Hiring Team,

I'm applying for the AI Red Team Analyst role. AI safety red-teaming isn't adjacent to my work — it *is* my work. My primary research project this year built the exact system you need someone to run.

**Adversarial LLM evaluation, at scale.** In *AI Safety Red-Team Evaluation* I built a dual-stage framework — LLM ensemble annotation (GPT-4o, Claude-3.5, Llama-3.2) feeding a Stacking Classifier — that scored **12,500 AI response pairs** across **6 harm categories** (dangerous info, hate, deception, privacy, illegal activity, self-harm) at **96.8% accuracy** with **Krippendorff's α = 0.81**. The classifier's precision was 97.2% and recall 96.1% (ROC-AUC 0.9923) at 850 samples/hour throughput.

**Attack taxonomy and defense analysis.** I used an 8-category MITRE ATLAS-aligned adversarial taxonomy and identified **multi-turn escalation as the highest-risk vector (31.8% of successful attacks)**. On the defense side, I quantified a dual-filter policy reducing raw harm rate from 21.8% to 4.8% — a 78% reduction with uncertainty bounds.

**Statistical honesty.** Bayesian hierarchical modeling with 95% HDI gave calibrated per-model vulnerability estimates (rather than the fragile point comparisons that a lot of safety benchmarks lean on). I also engineered 47 linguistic, semantic, and structural features so the classifier's decisions are auditable, not opaque.

**Compliance-ready.** Every artifact ships with a model card, SHAP explanations, and audit trails aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act — which becomes fully enforceable August 2, 2026, and is about to make automated red-teaming a documentation requirement, not a nice-to-have.

I'm authorized to work in the US and available for the posted remote schedule. I'd bring an operator's mindset, not just a research mindset.

Sincerely,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux | GitHub: https://github.com/dl1413

---

## Talking points for phone screen

- **Portfolio piece to send:** [AI Safety Red-Team Publication PDF](../../AI_Safety_RedTeam_Evaluation_Publication.pdf)
- **EU AI Act:** ready to speak to Article 15 red-team requirements landing Aug 2, 2026.
- **Multi-turn escalation:** the vector most teams under-invest in — I have the data.
