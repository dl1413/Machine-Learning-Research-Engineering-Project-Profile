# OpenAI — Researcher, Automated Red Teaming

**Posting:** https://openai.com/careers/researcher-automated-red-teaming-san-francisco/
**Location:** San Francisco (relocation or remote considered)
**Date drafted:** 2026-07-08

---

Dear OpenAI Automated Red Teaming team,

I'm applying for the Researcher role on Automated Red Teaming. The posting asks for strong applied-research instincts around evaluations, hands-on LLM and agent experience, and the ability to build scalable red-teaming automation — that's a precise description of the pipeline I shipped this year.

**Automated red-teaming, end-to-end:**

- **Dual-stage architecture (LLM ensemble → ML classifier).** 12,500 prompt/response pairs across 6 harm categories. Krippendorff's α = 0.81 across GPT-4o, Claude-3.5, and Llama-3.2 gave me a reliability floor to trust ensemble labels; a Stacking Classifier trained on 47 engineered features (linguistic, semantic, structural) then hit **96.8% accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923** at 850 samples/hour and $0.018/sample. Cost/coverage that a purely human loop cannot match — same delta you're trying to close on frontier models.
- **Adversarial taxonomy + defense measurement.** 8-category MITRE ATLAS-aligned attack taxonomy; multi-turn escalation surfaced as the highest-risk vector (31.8% of confirmed harms). Layered a dual-filter defense and measured **21.8% → 4.8% harm rate (78% relative reduction)** on a held-out set — the "generate attacks, measure defenses, iterate" loop your team runs.
- **Uncertainty done right.** Bayesian hierarchical model (PyMC, R-hat < 1.01, 95% HDI) over model-level risk so we report *credible* differences between models, not raw win rates — reduces the "which failure is real?" question to a statistic reviewers can defend.
- **Scalable engineering.** MLflow tracking, circuit breakers, adaptive rate-limiting, SHAP explanations for every prediction, audit trails aligned with IEEE 2830-2025 — so a finding survives the trip from a notebook to a policy readout.

Adjacent projects: an LLM ensemble bias-detection system (67,500 ratings, Friedman χ² = 42.73, p < 0.001) and a calibrated clinical ML system (99.12% acc, Platt-calibrated ECE 0.0089) — both reinforce the calibration + judge-ensemble skill set for automated evaluation.

Portfolio and code: https://dl1413.github.io/LLM-Portfolio/ • https://github.com/dl1413. Resume attached.

I'd welcome a conversation.

Best,
Derek Lankeaux
