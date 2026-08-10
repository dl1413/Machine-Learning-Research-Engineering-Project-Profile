# Cover Letter — 10a Labs, AI Red Teamer (Entry Level)

**Role:** AI Red Teamer (Entry Level) — Remote, United States
**Apply:** https://job-boards.greenhouse.io/10alabs/jobs/4002004009
**Lead projects:** P1 (AI Safety Red-Team Evaluation), P2 (LLM Ensemble Bias Detection)

---

Dear 10a Labs Hiring Team,

I'm applying for the AI Red Teamer (Entry Level) position. Your JD — adversarial test suites for LLMs and multimodal models, multilingual jailbreaks, escalation chains, vulnerability triage — is a near 1:1 match for what I spent the last six months building independently.

My **AI Safety Red-Team Evaluation** project ran 12,500 AI response pairs through a dual-stage GPT-4o / Claude-3.5 / Llama-3.2 ensemble across six harm categories (dangerous info, hate, deception, privacy, illegal activity, self-harm). The harness reached 96.8% classification accuracy with a Stacking Classifier (97.2% precision, 96.1% recall, ROC-AUC 0.9923) at 850 samples/hour, and the LLM annotators themselves held Krippendorff's α = 0.81 — the audit-grade reliability your client work depends on. The economics matter too: $0.018/sample vs. $6.12 for human annotation, a 340× cost reduction that's reusable across new policy edge cases.

Most relevant to your day-to-day red-teaming:

- **8-category MITRE ATLAS-aligned attack taxonomy** — multi-turn escalation surfaced as the highest-risk vector at 31.8%, the exact failure mode entry-level red teamers are usually asked to enumerate.
- **Defense-effectiveness analysis** — dual-filter reduced harm rate 21.8% → 4.8% (78% reduction), with the methodology written up in a publication-grade report.
- **47 engineered features** spanning linguistic, semantic, and structural harm signals — the kind of detector logic that backs a prompt library or scenario generator.

I'm comfortable scripting tests in Python (and the surrounding plumbing: circuit breakers, exponential backoff, MLflow tracking — I processed 80K API calls / 2.5M tokens on a parallel LLM-ensemble bias-detection project with the same infra). Bayesian hierarchical modeling in PyMC for multi-model risk quantification (95% HDI) on top of that.

Three publication-grade reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act — happy to share PDFs alongside this application.

Eager to bring this work to a team scaling adversarial testing across real client systems. Available immediately for interviews.

Best,
Derek Lankeaux, MS (Applied Statistics, RIT — 2026)
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
