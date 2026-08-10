# Cover Letter — Scale AI (Applied Scientist, LLM Evaluation / SEAL)

Dear Scale SEAL Team,

Scale's leaderboards are the reason my three independent projects converged on the same question: how do you measure model behavior credibly enough that the ranking survives scrutiny? I want to bring that instinct to SEAL.

In my **AI Safety Red-Team Evaluation** framework I designed two rubrics from scratch — a 6-category harm taxonomy and an 8-category MITRE-ATLAS-aligned adversarial taxonomy — and I ran them across GPT-4o, Claude-3.5, and Llama-3.2 on 12,500 response pairs. The ensemble held Krippendorff's α = 0.81 (audit-grade), fed a Stacking Classifier at 96.8% accuracy and 97.2% precision, and produced a per-model risk decomposition using PyMC hierarchical modeling (95% HDI). The finding that multi-turn escalation is the dominant attack vector (31.8% of confirmed harm) is the kind of eval-derived insight that changes how a leaderboard should be built.

My **LLM Ensemble Bias Detection** project is my proof that LLM-as-judge can be calibrated against contested ground truth. On 67,500 ratings across 4,500 textbook passages, PyMC partial pooling reached MCMC R-hat < 1.01, the Friedman test hit χ² = 42.73 (p < 0.001), and bootstrap CIs flagged the 12.3% highest-uncertainty passages for human review — the human-in-the-loop safety net every leaderboard needs. Ensemble pairwise correlation was 92%; the residual disagreement was the interesting signal, not noise.

My **Breast Cancer Classification** work (99.12% acc, ECE 0.0089 post-Platt) shows I take probability calibration seriously — critical when a rubric score gets aggregated into a leaderboard rank.

I have an Applied Statistics MS (RIT, 2026), am US-work-authorized, and I am fully remote-flexible with a preference for NYC or SF. I would welcome a conversation about the SEAL team or the broader Applied Science org.

Sincerely,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) • [GitHub](https://github.com/dl1413) • [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
