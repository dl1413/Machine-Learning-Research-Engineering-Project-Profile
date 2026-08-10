# Capital One — Principal Associate, Data Scientist — LLM Customization Team

**Location:** New York, NY (also posted in other Capital One hubs) • **Fit:** ⭐⭐ Good
**Apply:** https://www.capitalonecareers.com/job/new-york/principal-associate-data-scientist-llm-customization-team/1732/92083762528

**Why this fits:** "Principal Associate" is Capital One's early-mid IC level — MS-in-Applied-Stats candidates fit the profile, especially with production LLM work. Their LLM Customization Team focuses on adapting and evaluating LLMs for business applications — exactly the pattern of my LLM Bias Detection project (multi-model evaluation, Bayesian inference, production API pipelines at 2.5M+ tokens).

**Lead project:** LLM Ensemble Textbook Bias Detection
**Supporting:** AI Safety Red-Team (LLM evaluation infra + safety framing that Capital One cares about for regulated domains), Breast Cancer (MLOps + calibration rigor)

---

## Cover Letter

Dear Capital One Team,

I'm applying to the Principal Associate role on the LLM Customization Team because Capital One is one of the few large regulated organizations where responsible LLM adaptation is treated as a serious statistical problem — not just prompt tuning. That framing is where I want to spend my career.

My most relevant project is an LLM Ensemble Textbook Bias Detection system. I processed 67,500 bias ratings across 4,500 passages and 2.5 million tokens through a three-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2), reached 92% pairwise correlation across models with Krippendorff's α = 0.84 inter-rater reliability, and modeled publisher-level bias with a PyMC hierarchical model using partial pooling. MCMC converged cleanly (R-hat < 1.01), the Friedman test came in at χ² = 42.73 (p < 0.001), and 95% HDIs surfaced credible bias in 3 of 5 publishers. Production side: circuit breakers, exponential backoff, MLflow tracking — the stack that keeps an API-heavy pipeline reliable at scale.

That project maps directly to LLM customization work: multi-model evaluation to compare a customized model against a baseline, Bayesian hierarchical modeling to separate real segment-level differences from noise, and offline benchmarking infrastructure that a regulated organization can actually audit. My AI Safety Red-Team framework (dual-stage LLM ensemble, MITRE ATLAS taxonomy, 340× cost reduction over human annotation) shows the same pattern applied to safety evaluation — a lens Capital One's LLM work has to pass through.

I'm finishing my MS in Applied Statistics at RIT (2026) and I'd love to bring this toolkit to the LLM Customization Team.

Best,
Derek Lankeaux

---

## Resume-Bullet Variant

- Built multi-LLM evaluation framework (GPT-4o, Claude-3.5, Llama-3.2) processing 67,500 bias ratings across 4,500 passages and 2.5M tokens; 92% pairwise correlation, Krippendorff's α = 0.84 (excellent)
- Modeled publisher-level bias with PyMC hierarchical model (partial pooling, MCMC R-hat < 1.01); Friedman χ² = 42.73 (p < 0.001), 95% HDI surfaced credible bias in 3/5 publishers
- Engineered production LLM API pipeline: circuit breakers, exponential backoff, MLflow experiment tracking; processed 80K+ API calls with monitored drift and reproducible artifacts
- Aligned deliverables to IEEE 2830-2025 (Transparent ML) and ISO/IEC 23894:2025 (AI Risk Management) — the compliance surface regulated LLM work has to pass through

---

## 60-Second Hook

"I built a three-LLM evaluation framework — GPT-4o, Claude-3.5, Llama-3.2 — that processed 67,500 bias ratings across 4,500 passages, hit inter-rater reliability of α = 0.84 across the ensemble, and quantified real segment-level bias with a Bayesian hierarchical model that converged with R-hat under 1.01. That's the exact pattern LLM customization needs at Capital One: multi-model comparison, Bayesian uncertainty, production API infrastructure, and IEEE 2830-2025 audit trails. I'm finishing my MS in Applied Statistics and ready to apply that stack to your LLM work."
