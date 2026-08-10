# Cover Letter — Anthropic (Applied AI Engineer / Research Engineer, Evaluations)

Dear Anthropic Hiring Team,

I am applying to Anthropic's Evaluations team because the three research projects I have shipped over the past year converge on the exact problem you are staffing for: measuring frontier-model behavior with enough rigor that a safety case can rest on the result.

In my **AI Safety Red-Team Evaluation** framework, I ran 12,500 model-response pairs through a dual-stage pipeline — a GPT-4o / Claude-3.5 / Llama-3.2 ensemble producing structured harm annotations, feeding a Stacking Classifier that reached 96.8% accuracy and 97.2% precision across six harm categories. The ensemble held audit-grade reliability (Krippendorff's α = 0.81) while cutting cost from $6.12 per human-annotated sample to $0.018 — a 340× reduction at 850 samples/hour. I built the adversarial taxonomy against MITRE ATLAS categories and used Bayesian hierarchical modeling with a 95% HDI to quantify per-model vulnerability. The finding that multi-turn escalation is the dominant harm vector (31.8% of confirmed incidents) is exactly the kind of eval-derived insight that should drive RSP thresholds.

My **LLM Ensemble Textbook Bias Detection** project pushed the same eval methodology into a domain where ground truth is contested. Across 4,500 passages, three frontier LLMs produced 67,500 ratings; PyMC hierarchical partial pooling reached MCMC convergence (R-hat < 1.01), and the Friedman test (χ² = 42.73, p < 0.001) flagged statistically credible bias in 3 of 5 publishers. Passage-level bootstrap CIs flagged the 12.3% highest-uncertainty items for expert review — the human-in-the-loop routing pattern I would expect a safety eval pipeline to need.

My **Breast Cancer Classification** project (99.12% acc, 100% precision, ECE 0.0089 after Platt scaling) is my proof that I take calibration seriously — the same discipline you need when a threshold decision on an eval score determines whether a model ships.

I have an Applied Statistics MS (RIT, expected 2026) and am authorized to work in the US. I am available in NYC, hybrid, or fully remote. I would welcome a conversation about the Evaluations, Frontier Red Team, or Applied AI Engineer families.

Sincerely,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) • [GitHub](https://github.com/dl1413) • [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
