# Cover Letter — Disney, Senior Data Scientist — Experimentation & Causal Inference

**Role:** Senior Data Scientist, Experimentation and Causal Inference — Direct to Consumer
**Location:** New York, NY
**Apply:** https://jobs.disneycareers.com/job/new-york/senior-data-scientist-experimentation-and-causal-inference/391/89132547520
**Lead projects:** P2 (LLM Ensemble Bias Detection), P3 (Breast Cancer ML Classification)

> Note: This role is posted as Senior. Derek is finishing his MS in 2026 — frame the level honestly in the application and let the hiring team gauge fit. The statistical depth in P2/P3 is genuinely senior-shaped work.

---

Dear Disney Direct to Consumer Hiring Team,

I'm applying for the Senior Data Scientist, Experimentation & Causal Inference role. The team's charter — architecting experiments across the subscriber journey from acquisition through retention and revenue — is the kind of measurement work I've structured both of my recent research projects around.

**On the experimentation and inference side**, my **LLM Ensemble Textbook Bias Detection** project ran what is, structurally, a multi-arm comparison: five publishers × three LLM "raters" × 4,500 passages = 67,500 ratings. I designed it around the same questions a subscriber-journey experiment asks:

- *Is the effect real?* Friedman χ² = 42.73, p < 0.001, with Bonferroni / FDR correction on the publisher comparisons.
- *How big is it, and how sure am I?* Bayesian hierarchical model with partial pooling in PyMC, MCMC convergence at R-hat < 1.01, 95% HDI on each publisher's bias estimate, bootstrap CIs on the per-passage uncertainties (12.3% flagged as high-uncertainty and routed to expert review).
- *Are my measurements themselves reliable?* Inter-rater Krippendorff's α = 0.84 across three frontier LLMs (92% pairwise correlation). This is the same instrument-validity question A/B testing usually skips.

**On the modeling and decision-policy side**, my **Breast Cancer ML Classification** system gets at the operational counterpart of experimentation: once you have a result, what threshold do you act on? The system reached 99.12% accuracy with Platt-calibrated probabilities (ECE reduced 71.5%, 0.0312 → 0.0089) and context-adaptive thresholds — 100% sensitivity at 0.31 for screening, a different threshold for confirmation. That logic ports directly to subscriber-journey decisions where the cost of a false retain vs. a false churn-intervention are different numbers.

I am MS Applied Statistics (RIT, expected 2026), so I want to be transparent: my paid full-time experience is shorter than a typical senior. The statistical depth and end-to-end ownership shown in three publication-grade reports — IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act aligned — is where I'd ask the team to weight my candidacy. If Disney is open to a level discussion, I'd welcome the conversation.

Best,
Derek Lankeaux, MS (Applied Statistics, RIT — 2026)
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
