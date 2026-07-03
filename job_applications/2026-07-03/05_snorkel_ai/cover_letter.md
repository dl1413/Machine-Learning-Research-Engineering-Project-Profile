# Cover Letter — Snorkel AI (Applied ML Data Scientist)

Dear Snorkel Team,

Snorkel's bet — that a stack of noisy, cheap label sources can be modeled into a clean training signal — is the exact architecture I have built in three consecutive research projects, without knowing at the time that I was reinventing Snorkel's labeling-function paradigm.

My **LLM Ensemble Bias Detection** framework treats each of three frontier LLMs (GPT-4o, Claude-3.5, Llama-3.2) as a noisy labeler over 4,500 textbook passages, yielding 67,500 ratings. Pairwise correlation was 92%, ensemble Krippendorff's α = 0.84 — but the interesting move is what I did with the residual disagreement. PyMC hierarchical partial pooling recovered per-publisher bias credible intervals (95% HDI, MCMC R-hat < 1.01) and passage-level bootstrap CIs flagged the 12.3% highest-uncertainty items for human review. That is the labeling-function → generative-model → uncertainty-aware routing loop Snorkel productizes.

My **AI Safety Red-Team Evaluation** framework is the same architecture, pushed to production throughput: 12,500 model responses, 850 samples/hour, $0.018 per sample (340× cheaper than a human baseline), 80K+ API calls with circuit breakers and exponential backoff. The dual-stage design (LLM ensemble labeling → Stacking Classifier at 96.8% accuracy) is a working proof that LFs-plus-downstream-model outperforms either alone.

My **Breast Cancer** classifier is where I proved I take calibration seriously — Platt scaling drove Expected Calibration Error from 0.0312 to 0.0089, a 71.5% reduction. Label quality is worthless if downstream probability isn't calibrated.

I have an Applied Statistics MS (RIT, 2026), am US-work-authorized, and I am fully remote. I would welcome a conversation about Snorkel's Applied ML Data Scientist or Research Scientist tracks.

Sincerely,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) • [GitHub](https://github.com/dl1413) • [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
