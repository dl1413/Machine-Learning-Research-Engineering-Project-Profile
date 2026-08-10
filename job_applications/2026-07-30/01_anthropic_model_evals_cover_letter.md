# Cover Letter — Anthropic, Research Engineer (Model Evaluations)

**Role:** Research Engineer, Model Evaluations
**Locations:** San Francisco, CA · New York, NY
**Apply:** https://www.anthropic.com/careers
**Lead project:** AI Safety Red-Team Evaluation Framework

---

Dear Anthropic Model Evaluations team,

The role — designing evaluation methodologies across reasoning, safety, helpfulness, and harmlessness, and building the high-throughput evaluation infrastructure that runs during production training — is exactly the work I've been doing independently and would want to do at a frontier lab.

My most relevant project is an AI Safety Red-Team Evaluation framework I built and published in April 2026. It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking meta-classifier on 47 harm-signal features across 6 harm categories (dangerous info, hate, deception, privacy, illegal activity, self-harm). Against a 12,500 response-pair benchmark it reaches **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**, with inter-rater reliability at **Krippendorff's α = 0.81**. The pipeline runs at **850 samples/hour for $0.018/sample — a 340× cost reduction versus human annotation** — with circuit breakers, exponential backoff, MLflow lineage, and SHAP-based explanations for every classification. I shipped it under IEEE 2830-2025 audit-trail requirements.

Two things I'd bring to your evaluation platform beyond the throughput number:

1. **Judge disagreement is a signal, not noise.** I fit a PyMC Bayesian hierarchical model over the three LLM judges (partial pooling, MCMC R-hat < 1.01) to produce **95% HDI risk intervals per model family**, surfacing where GPT-4o and Claude systematically diverge — the same instrument you'd want when a new capability appears mid-training and one judge lights up first.

2. **Multi-turn escalation as the top risk vector.** My adversarial taxonomy (8 MITRE-ATLAS-aligned categories) flagged multi-turn escalation as the highest-impact vector (31.8% of confirmed harms) and let a dual-filter defense reduce measured harm rate 21.8% → 4.8%. That structured red-team library is directly transferable.

I've published two adjacent projects — a multi-LLM bias detection study (67,500 ratings, α = 0.84, Friedman χ² = 42.73, p < 0.001) and a clinical-grade classifier (99.12% accuracy, ROC-AUC 0.9987) — that show the same eval-first habit in other domains.

I'm finishing an MS in Applied Statistics at RIT (Bayesian methods, ML, experimental design), targeting a 2026 start, based in / available for New York or remote, US-authorized. Portfolio, code, and full technical reports: github.com/dl1413. Happy to walk through the red-team eval — it's the fastest way to see how I'd think about Anthropic's evaluation problems.

Best,
Derek Lankeaux
