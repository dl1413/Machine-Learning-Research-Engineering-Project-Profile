# Etsy — Senior Applied Scientist, Trust & Safety

**Location:** New York, NY / Brooklyn (hybrid)
**Apply (careers hub — filter for "Trust & Safety"):** https://www.dsml.etsy.com/
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting projects:** LLM Bias Detection (Bayesian judge-disagreement for content policy), Breast Cancer Classification (calibrated probabilities → enforcement thresholds)

---

Dear Etsy Trust & Safety team,

Trust & Safety at Etsy's scale is a classification problem where the model's calibration and the taxonomy behind it matter as much as the top-line accuracy — because the cost of a false positive to a small seller is very different from a false negative on genuinely harmful content. That framing is exactly what my AI Safety Red-Team Evaluation Framework is built around, and it's why I'm applying for the Senior Applied Scientist role.

I published the Red-Team framework in April 2026 as an independent research project. It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as harm-detection judges on 12,500 response pairs across a 6-category taxonomy (dangerous info, hate, deception, privacy, illegal activity, self-harm) and trains a 47-feature stacking meta-classifier that hits **96.8% accuracy, 97.2% precision, 96.1% recall, and ROC-AUC 0.9923**. Krippendorff's α = 0.81 across the three judges shows the inter-model reliability holds at the audit-grade line. The system runs at 850 samples/hour for $0.018/sample — a 340× cost reduction versus human annotation — with circuit breakers, exponential backoff, and MLflow lineage baked in. Trust & Safety teams live and die on throughput at review time, so operational hygiene was a design constraint, not an afterthought.

Two pieces map particularly closely to what Etsy's T&S team faces:

1. **Defense-effectiveness analysis.** I paired the detector with a dual-filter defense that reduced harm rate from 21.8% to 4.8% (78% reduction), then quantified marginal contribution per filter. Same pattern applies to layered enforcement in a marketplace: keyword filter → ML detector → human review → appeal.
2. **Adversarial taxonomy.** I built an 8-category MITRE ATLAS-aligned attack taxonomy and identified multi-turn escalation (31.8% share) as the highest-risk vector. That's the shape of the harm surface Etsy Trust & Safety has to reason about across listings, messages, and reviews.

To close the loop on trust, I added a PyMC Bayesian hierarchical model on top of the judge outputs to produce 95% HDIs per judge — so a T&S policy analyst can say "we're 95% confident this listing sits above threshold X" rather than "the model thinks it's bad." The whole system is documented under IEEE 2830-2025 audit trails.

I recognize this is a Senior title and I'm early-career (MS Applied Statistics, RIT 2026) — I'm putting the portfolio forward on the theory that the shipped depth compensates for years-of-experience count. Happy to walk through the report end-to-end.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
