# AirOps — Data Scientist / MLE

**Posting:** https://jobs.ashbyhq.com/airops/8fec67ba-1d12-4ecb-b1dc-610d28f85a51
**Location:** Remote / NYC
**Why fit:** AirOps builds ML systems on top of AI search behavior — search/rec models plus LLM-driven evaluation. Multi-LLM ensembles and Bayesian credible-bias detection at production scale are exactly my track.

---

Dear AirOps team,

I'm applying for the Data Scientist / MLE role. Designing end-to-end ML systems on AI search behavior — NLP, ranking, evaluation — is what I've shipped, and three production-style projects show it.

**LLM Ensemble Textbook Bias Detection** is the closest analog to "analyze AI search behavior." 4,500 passages, 67,500 ratings, three frontier LLMs (GPT-4o, Claude-3.5, Llama-3.2) with 92% pairwise correlation and Krippendorff's α = 0.84. The PyMC hierarchical model with partial pooling (R-hat < 1.01) returned publisher-level credible bias (Friedman χ² = 42.73, p < 0.001) — the same statistical machinery I'd use to attribute ranking shifts to publisher / domain / topic effects with proper uncertainty, not just point estimates. Bootstrap CIs flagged 12.3% of passages as high-uncertainty for routing — analogous to confidence-gating which queries get the expensive LLM versus a cached score.

**AI Safety Red-Team Evaluation** is the production pattern: LLM ensemble labels → Stacking Classifier on 47 engineered features → 96.8% accuracy at 850 samples/hour and $0.018 / sample (340× cheaper than human). Circuit breakers, exponential backoff, MLflow tracking — pipelines that don't fall over under 80K+ API calls.

**Breast Cancer Classification** shows the ranking / threshold discipline: Optuna TPE (45 trials vs 240 grid), Platt-calibrated probabilities (ECE 0.0089), context-specific thresholds, FastAPI under 100ms p95.

I'd love to bring this stack to AirOps. Portfolio: https://dl1413.github.io/LLM-Portfolio/

Best,
Derek Lankeaux
