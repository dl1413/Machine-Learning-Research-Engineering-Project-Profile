# Anthropic — Research Engineer / Scientist, Frontier Red Team (Cyber)

**Posting:** https://job-boards.greenhouse.io/anthropic/jobs/5076477008
**Location:** SF / NYC / remote-friendly
**Date drafted:** 2026-07-08

---

Dear Anthropic Frontier Red Team,

I'm applying for the Research Engineer / Scientist role on the Frontier Red Team (Cyber). Anthropic's public thesis — that 2026 is the year models cross expert thresholds in several cyber domains, and that red-teaming is how we stay ahead of it — is exactly the problem I've been building toward. I recently shipped a red-teaming pipeline that scales the parts of harm evaluation that humans can't sustainably do alone, and I'd like to bring that work to your team.

**Directly relevant work:**

- **AI Safety Red-Team Evaluation Framework (2026).** I designed a dual-stage LLM-ensemble → ML-classifier pipeline for automated harm detection: 12,500 prompt/response pairs across 6 harm categories, 96.8% accuracy (97.2% precision / 96.1% recall) with a Stacking Classifier over 47 engineered linguistic-semantic-structural features. The ensemble (GPT-4o + Claude-3.5 + Llama-3.2) hit Krippendorff's α = 0.81 against a human-labeled subset, giving auditable inter-rater reliability at **$0.018/sample vs $6.12 for human annotation — a 340× cost reduction** at 850 samples/hour. Multi-turn escalation was the highest-risk vector (31.8% of confirmed harms); a dual-filter defense dropped harm rate from 21.8% → 4.8%.
- **LLM Ensemble Bias Detection (2026).** I built a Bayesian hierarchical model with partial pooling (PyMC, MCMC R-hat < 1.01, 95% HDI) over 67,500 ratings from a multi-LLM ensemble — the same statistical machinery I'd use to quantify model-level cyber capability with defensible uncertainty rather than a point estimate.
- **Production pipeline hygiene.** All of the above ran through MLflow tracking, circuit breakers, exponential backoff, and SHAP audit trails aligned with IEEE 2830-2025 and ISO/IEC 23894:2025 — the kind of reproducibility Frontier Red Team publications require.

**Why the Cyber team specifically:** the framework I already built generalizes cleanly to cyber-specific harm taxonomies (MITRE ATT&CK / ATLAS rather than the safety taxonomy I used) and to multi-turn autonomous agent traces rather than single-turn dialogue. I'd expect the first month to be re-scoping the classifier and feature set to your existing evaluation harness and adding capability-elicitation prompts for cyber tasks.

Publications and code samples: https://github.com/dl1413 and https://dl1413.github.io/LLM-Portfolio/. Resume attached.

Thank you for considering my application.

Best,
Derek Lankeaux
