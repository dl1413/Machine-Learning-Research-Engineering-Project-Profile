# Cover Letter — Dataminr, Senior Research Scientist (NLP, LLM, GenAI)

Dear Dataminr AI Research Team,

I'm applying for the Senior Research Scientist role on NLP, LLM, and GenAI. Dataminr's product depends on catching real signals at high precision inside a firehose of noisy language — and that shape of problem is exactly what my three research projects converge on.

My AI Safety Red-Team Evaluation Framework is the closest structural analogue. It combines an LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) as an annotation layer with a Stacking Classifier over 47 engineered linguistic / semantic / structural features, scoring 12,500 responses across 6 harm categories at 97.2% precision, 96.1% recall, and ROC-AUC 0.9923. Just as important for a low-false-positive product: I built the classifier around a MITRE ATLAS-aligned adversarial-attack taxonomy — 8 vectors, of which multi-turn escalation surfaced as the highest-risk pathway at 31.8% success — and validated the defense side end-to-end (dual-filter defense reduces harm rate 21.8% → 4.8%). The cost story is what makes this a research-to-production system rather than a benchmark: $0.018/sample versus $6.12 human annotation, a 340× reduction, with Krippendorff's α = 0.81 preserved. That's the pattern any high-frequency detection product needs to run its evaluations at the same cadence as its inference.

The research bar shows up in my other two projects. My LLM Ensemble Bias Detection system is a Bayesian hierarchical study across 4,500 passages and 67,500 ratings — PyMC partial pooling with R-hat < 1.01, 95% HDI at publisher and topic layers, Friedman χ² = 42.73 (p < 0.001), and a Spearman correlation matrix (ρ up to 0.74) revealing structural editorial relationships. My breast-cancer classifier project pushes the same statistical rigor into the calibration and threshold-policy space (99.12% accuracy, 100% precision, Platt calibration to ECE 0.0089, FastAPI < 100 ms p95). All three shipped as publication-grade technical reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act — writing samples I'd be happy to walk you through.

The reason I'd want to be at Dataminr specifically: the real-time constraint changes what "good evaluation" means. Point-estimate accuracy on a benchmark isn't enough — you need calibrated confidence per event, drift-aware monitoring, and the ability to defend a positive to a customer under scrutiny. That's the intersection of applied statistics and production ML I've been building around, and I'd love the chance to do it against a live data stream.

Warmly,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
