# Cover Letter — Lakera, AI Security Engineer — Red Team

**Role:** AI Security Engineer — Red Team (United States, Remote)
**Apply:** https://jobs.ashbyhq.com/lakera.ai/75dd7f97-8ae2-460f-ae7e-cfceade9c1c6
**Lead projects:** P1 (AI Safety Red-Team Evaluation), P2 (LLM Ensemble Bias Detection)

---

Dear Lakera Hiring Team,

I'm writing to apply for the AI Security Engineer — Red Team role. The combination you describe — hands-on red teaming plus building frameworks, methodologies, and tooling that scale a services org — is exactly the problem I spent the last two semesters solving for myself, just at smaller scale.

My **AI Safety Red-Team Evaluation Framework** is, in effect, a methodology + tooling stack for repeatable adversarial testing of frontier LLMs:

- **Dual-stage pipeline**: LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) for annotation → Stacking Classifier for triage. 96.8% accuracy across 12,500 response pairs, six harm categories, Krippendorff's α = 0.81.
- **8-category MITRE ATLAS-aligned attack taxonomy** with multi-turn escalation identified as the dominant risk vector (31.8%). This is the kind of structured-output a delivery team can hand a client.
- **Defense-effectiveness measurement**: dual-filter reduced harm rate 21.8% → 4.8% — a quantified before/after that survives client scrutiny.
- **47 engineered features** (linguistic, semantic, structural) feeding the classifier — the seed of a prompt library or detector registry.
- **Production engineering**: circuit breakers, exponential backoff, MLflow experiment tracking, 850 samples/hour throughput at $0.018/sample (340× cheaper than human annotation).
- **Bayesian hierarchical modeling** for multi-model risk quantification with 95% HDI — useful when a client asks "how confident are you that model X is safer than model Y?"

I deliberately built this for reuse: another project, **LLM Ensemble Bias Detection**, reuses the same multi-LLM scaffold (92% pairwise correlation, R-hat < 1.01, Friedman χ² = 42.73, p < 0.001) to detect bias in 4,500 textbook passages — proof the framework generalizes beyond harm taxonomy.

Both projects are documented as publication-grade technical reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act. PDFs and code references available on request.

I would bring the same instinct to Lakera: ship the tooling alongside the engagement, so the second client is faster than the first. Happy to walk through the harm-rate-reduction analysis or the attack taxonomy on a first call.

Best,
Derek Lankeaux, MS (Applied Statistics, RIT — 2026)
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
