# TikTok — Data Scientist, Analytics (Trust & Safety)

**Location:** New York, NY · **Apply:** https://careers.tiktok.com/ (search "Data
Scientist Trust and Safety New York") · **Fit:** ★★★★★

---

Dear TikTok Trust & Safety Analytics team,

I'm applying for the Data Scientist, Analytics role on Trust & Safety in New York.
The description — "measuring risks, finding insights, and diagnosing problems with
data" alongside product, engineering, and policy — is exactly the seat I want to sit
in, and I've already shipped the work to prove it.

Two of my three published projects are directly about T&S measurement at platform
scale. My **AI Safety Red-Team Evaluation** pipeline scores 12,500 responses across
six harm categories (dangerous info, hate, deception, privacy, illegal activity,
self-harm) using a multi-LLM ensemble and a 47-feature Stacking Classifier —
**96.8% accuracy, 850 samples/hour, $0.018/sample (340× cheaper than human review),
Krippendorff's α = 0.81**. The multi-turn adversarial taxonomy identified escalation
as the highest-risk pattern at 31.8%, and the dual-filter defense I evaluated cut
harm rate from 21.8% to 4.8%. Everything is SHAP-explained and audit-logged.

My **LLM Ensemble Bias Detection** project applies the same pattern to a fairness
question: 67,500 ratings across 4,500 pieces of content, three frontier LLMs at 92%
pairwise correlation, and a **PyMC hierarchical model (R-hat < 1.01, 95% HDI)** that
flagged 3/5 publishers with statistically significant bias (Friedman χ² = 42.73,
p < 0.001) and 12.3% of passages for human review via bootstrap uncertainty.

The DS toolkit T&S actually uses — SQL, A/B testing, multiple-testing correction
(Bonferroni/FDR), effect-size reporting, Bayesian hierarchical modeling for
publisher/creator-level effects, and stakeholder-ready model cards — is what I've
been practicing on real datasets, not just coursework. I'd love to talk about
measurement design and how the ensemble-plus-classifier pattern could plug into your
policy workflows.

Derek Lankeaux · [LinkedIn](https://linkedin.com/in/derek-lankeaux)
· [GitHub](https://github.com/dl1413)
