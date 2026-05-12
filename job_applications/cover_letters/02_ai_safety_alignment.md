# Cover Letter Template — AI Safety / Alignment / Trust & Safety Engineer

**Best fit for:** Anthropic, OpenAI safety teams, DeepMind safety, Google Trust & Safety, Meta Responsible AI, Scale AI red-team, Cohere safety, Hugging Face evals, NYC AI policy orgs (AI Now, Partnership on AI), startup safety/eval companies.

**Anchor project:** AI Safety Red-Team Evaluation (lead heavily). Secondary: LLM Bias Detection.

---

Dear {{Hiring Manager Name | Safety Team}},

I'm writing to apply for the **{{Role Title}}** position at **{{Company}}**. {{Company}}'s work on {{specific safety artifact — e.g., Responsible Scaling Policy, model card, evaluation paper, jailbreak research}} is exactly the kind of empirical, measurable safety work I want to be building.

Over the past year I've built the **AI Safety Red-Team Evaluation Framework** — an end-to-end system for automated harm detection that I designed because I kept seeing safety evals either rely on small expert panels (slow, inconsistent) or single-model judges (biased, low reliability). My approach:

- **Dual-stage ensemble**: GPT-4o, Claude-3.5, and Llama-3.2 generate annotations, which are then aggregated by a Stacking Classifier over 47 engineered linguistic/semantic/structural features.
- **96.8% accuracy** on 12,500 response pairs across 6 harm categories (dangerous info, hate, deception, privacy, illegal activity, self-harm).
- **Krippendorff's α = 0.81** inter-rater reliability — i.e., the LLM panel agrees with itself at a level considered acceptable for published human-rater studies.
- **340× cost reduction** vs. human annotation ($0.018/sample vs $6.12) at 850 samples/hr.
- **MITRE ATLAS–aligned attack taxonomy** (8 categories); multi-turn escalation flagged as the highest-risk vector at 31.8% success rate.
- **Defense effectiveness analysis**: dual-filter reduces harm rate from 21.8% → 4.8% (78% reduction).
- **Bayesian hierarchical risk modeling** with 95% HDI uncertainty quantification, SHAP audit trails, and IEEE 2830-2025 / ISO/IEC 23894:2025 compliance.

I've also applied the same multi-LLM-with-Bayesian-pooling pattern to bias detection across 67,500 ratings of textbook content (α = 0.84, publisher-level credible bias with Friedman χ² = 42.73, p < 0.001), so the methodology generalizes beyond harm.

What I think I'd add at {{Company}}:

1. **Evals that are calibrated, not just accurate** — I default to reporting reliability coefficients, calibration error, and credible intervals, not just point accuracy.
2. **Adversarial thinking grounded in taxonomy** — I've spent enough time inside attack categorization (MITRE ATLAS, multi-turn, prompt injection variants) to design evals that stress what actually breaks models in deployment.
3. **Shippable infrastructure** — Circuit breakers, rate limiting, MLflow tracking, FastAPI serving — production hygiene from day one.

I'd love to discuss how this could plug into {{specific team / eval workstream}}. Reports and code: [github.com/dl1413](https://github.com/dl1413).

Thank you,

Derek Lankeaux
MS Applied Statistics, RIT (2026) · [LinkedIn](https://linkedin.com/in/derek-lankeaux) · US work auth · Remote / NYC

---

### Customization notes
- Reference one **specific** safety artifact (RSP, eval, model card, paper). Generic "your safety work" reads as mass-applied.
- If the role is policy/governance rather than engineering, swap point 3 ("Shippable infrastructure") for "Compliance-aware reporting" and lead with IEEE 2830-2025 / EU AI Act experience.
- For Anthropic specifically: reference Constitutional AI, the RSP, or a specific eval paper. The tech stack already includes Claude-3.5 — call it out by name.
