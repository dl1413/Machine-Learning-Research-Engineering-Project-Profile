# Innodata — Applied Research Scientist, LLM Evaluation & Post-Training

**Location:** Remote • **Fit:** ⭐ Stretch (posting asks for 5+ YOE / PhD preferred)
**Apply:** https://job-boards.greenhouse.io/innodatainc/jobs/4288050009

**Why this fits:** The exact shape of what they do — evaluation frameworks and methodologies for LLM and multimodal systems — is the shape of my LLM Bias Detection AND AI Safety Red-Team work. The YOE bar is a stretch for a 2026 MS grad, but applied research is a portfolio-driven hiring lane; a specific, technically deep pitch that maps my two LLM-eval projects to their published problems is worth the shot. Frame as "MS-early-career with two production LLM-eval systems already shipped; open to Research Assistant or Applied Scientist I-level if that's a better fit."

**Lead project:** LLM Ensemble Textbook Bias Detection
**Supporting:** AI Safety Red-Team Evaluation (evaluation design for safety-critical LLM outputs)

---

## Cover Letter

Dear Innodata Research Team,

I'm applying for the Applied Research Scientist role on the LLM Evaluation & Post-Training team. I know the posting asks for five-plus years and a PhD is preferred; I'm finishing my MS in Applied Statistics in 2026. What I want to make the case for is that I've already shipped two production LLM-evaluation systems — the exact class of research your team is doing — and I'm eager to keep pushing on evaluation design as an Applied Scientist I or Research Assistant if that's the better on-ramp.

The most on-thesis project: an LLM Ensemble Bias Detection system where I built a three-model evaluation framework (GPT-4o, Claude-3.5, Llama-3.2), processed 67,500 bias ratings across 4,500 textbook passages, and modeled the results with a PyMC hierarchical model using partial pooling. What I care about — and what maps directly to post-training research — is the measurement design: 92% pairwise correlation among the LLM raters, Krippendorff's α = 0.84 inter-rater reliability, MCMC that converged cleanly (R-hat < 1.01), and 95% HDIs that separated credible publisher-level bias (3 of 5 significant) from noise. Friedman χ² = 42.73 (p < 0.001) confirmed the effect. The point isn't the accuracy of any single model — it's that the evaluation framework itself is auditable, statistically rigorous, and reproducible.

The AI Safety Red-Team project is the same story applied to harm detection: a dual-stage evaluation pipeline (LLM ensemble → stacking classifier over 47 features) achieving 96.8% accuracy on 12,500 responses, a MITRE ATLAS-aligned taxonomy, and Bayesian hierarchical modeling for cross-model risk analysis — the design question of how feedback signals shape post-training is exactly what I've been thinking about.

I'd bring rigorous statistical grounding and shipped LLM-eval infrastructure. Happy to make the case in more detail.

Best,
Derek Lankeaux

---

## Resume-Bullet Variant

- Built two production LLM-evaluation frameworks (bias detection + safety red-teaming) processing 80K+ evaluations across GPT-4o, Claude-3.5, and Llama-3.2; both hit Krippendorff's α ≥ 0.81 inter-rater reliability
- Designed measurement infrastructure with PyMC hierarchical modeling, MCMC diagnostics (R-hat < 1.01), 95% HDIs, and Friedman χ² for cross-model significance testing — separating credible effects from noise
- Cut LLM-eval cost 340× ($0.018/sample vs $6.12 human baseline) at 850 samples/hr while preserving audit-grade reliability, an evaluation-cost curve directly relevant to post-training feedback loops
- Aligned deliverables to IEEE 2830-2025 and ISO/IEC 23894:2025 — evaluation frameworks that regulated and safety-critical LLM work can be audited against

---

## 60-Second Hook

"I know the posting asks for five years and prefers a PhD. I'm an MS candidate finishing in 2026 — but I've already shipped two production LLM-evaluation systems that map exactly to the work your team publishes: a three-LLM bias-detection framework with 67,500 ratings modeled through a PyMC hierarchical posterior, and an AI safety red-team pipeline evaluating 12,500 harm-labeled response pairs across six categories with a MITRE ATLAS-aligned taxonomy. If the seniority bar means I'd fit better as Applied Scientist I, I'd love that conversation."
