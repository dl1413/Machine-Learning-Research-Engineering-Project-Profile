# The New York Times — Machine Learning Scientist

**Posting:** https://job-boards.greenhouse.io/thenewyorktimes/jobs/4690108005
**Location:** New York, NY
**Why fit:** NYT's ML team works on recommendation, content understanding, and editorial quality measurement — bias detection, ensemble LLM judging, and reproducible technical reporting are exactly the things I have shipped.

---

Dear NYT ML Hiring Team,

I'm applying for the Machine Learning Scientist role. My MS in Applied Statistics at RIT (2026) and three production-style ML projects line up with NYT's mission of using ML to enrich journalism without compromising editorial trust.

My **LLM Ensemble Textbook Bias Detection System** is the most direct analog. I processed 67,500 bias ratings across 4,500 passages with a GPT-4o / Claude-3.5 / Llama-3.2 ensemble (92% pairwise correlation, Krippendorff's α = 0.84) and a PyMC hierarchical model with partial pooling. Friedman χ² = 42.73, p < 0.001 surfaced statistically credible bias in 3/5 publishers, with bootstrap CIs flagging 12.3% of passages as high-uncertainty and routing them to human review — the same human-in-the-loop pattern I'd want for editorial-content classifiers at NYT scale.

My **AI Safety Red-Team Evaluation** added a Stacking Classifier on 47 engineered features (96.8% accuracy) over the LLM-ensemble labels — a two-stage pattern that turns expensive LLM judgments into cheap, monitorable production scores. It cut per-sample cost from $6.12 to $0.018 (340×) while preserving α = 0.81.

My **Breast Cancer Classification** project shows the calibration discipline I bring to any classifier touching consequential decisions: ECE 0.0089 after Platt scaling and threshold tuning per decision context.

Across all three, I shipped publication-grade reports with model cards, SHAP explanations, and IEEE 2830-2025 / EU AI Act-aligned documentation — relevant for a newsroom that publishes its methodology.

I'd welcome a conversation. Portfolio: https://dl1413.github.io/LLM-Portfolio/

Best,
Derek Lankeaux
