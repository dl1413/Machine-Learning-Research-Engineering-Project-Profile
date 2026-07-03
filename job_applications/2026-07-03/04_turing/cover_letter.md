# Cover Letter — Turing (Applied Research Scientist, LLM Evaluation & Post-Training)

Dear Turing Research Team,

Turing's business is proving that post-training decisions — SFT curation, rubric-scored RLHF, RLVR — actually move a downstream capability number. That is exactly the question I have been answering in miniature across three independent research projects.

My **LLM Ensemble Bias Detection** framework processed 67,500 rubric-scored ratings across 4,500 passages using GPT-4o, Claude-3.5, and Llama-3.2 in ensemble (92% pairwise correlation, α = 0.84). I used PyMC hierarchical partial pooling to recover publisher-level bias credible intervals (95% HDI, MCMC R-hat < 1.01) and the Friedman test (χ² = 42.73, p < 0.001) confirmed the ranking was statistically credible. Bootstrap CIs at the passage level flagged the 12.3% highest-uncertainty items for expert relabeling — the same human-in-the-loop refinement loop RLHF pipelines rely on to keep signal-to-noise high.

My **AI Safety Red-Team Evaluation** framework is the same architecture pointed at rubric-scored safety evals. 12,500 model-response pairs, 6 harm categories, an 8-category adversarial taxonomy, and a Stacking Classifier at 96.8% accuracy (97.2% precision). Throughput of 850 samples/hour at $0.018 per sample (340× cheaper than human annotation) with 80K+ API calls managed by circuit breakers — the operational shape a Turing client would recognize.

My **Breast Cancer** classifier is where I proved calibration is not optional: Platt scaling drove Expected Calibration Error from 0.0312 to 0.0089. Rubric-scored eval loops need calibrated aggregation to avoid over-crediting confident-but-wrong judges.

I have an Applied Statistics MS (RIT, 2026), am US-work-authorized, and I am fully remote. I would welcome a conversation about the Applied Research Scientist or LLM Evaluation & Post-Training track.

Sincerely,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) • [GitHub](https://github.com/dl1413) • [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
