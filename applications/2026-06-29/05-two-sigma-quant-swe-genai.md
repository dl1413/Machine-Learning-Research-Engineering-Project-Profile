# Cover Letter — Two Sigma, Quantitative Software Engineer (Generative AI)

**Role:** Quantitative Software Engineer — Generative AI
**Location:** New York, NY
**Apply:** https://careers.twosigma.com/careers/JobDetail/New-York-City-United-States-Quantitative-Software-Engineer-Generative-AI/13079
**Lead projects:** P1 (AI Safety Red-Team), P3 (Breast Cancer Classification — for production engineering)

---

Dear Two Sigma News Engineering Team,

I'm applying for the Quantitative Software Engineer — Generative AI role. The JD describes work that sits at the seam of NLP / LLM services and production pipelines, and that's the shape of work I've been doing for the last six months on my own.

**Production LLM ensemble at scale.** My AI Safety Red-Team Evaluation project ran 12,500 AI response pairs through a three-model ensemble (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) and a Stacking Classifier on top — 96.8% accuracy, 850 samples/hour throughput, $0.018/sample. Same year I ran an LLM Ensemble Bias Detection pipeline that processed 67,500 ratings / 4,500 passages / 2.5M tokens with the same multi-model scaffold (Krippendorff's α = 0.84, R-hat < 1.01).

The pipeline plumbing is the part I think matches the JD most directly:

- **Resilient API integration**: circuit breakers, exponential backoff, adaptive rate limiting — 80K+ API calls executed without manual intervention.
- **Experiment tracking**: MLflow registry across both projects so every run is reproducible and comparable.
- **Low-latency serving**: my Breast Cancer Classification system uses FastAPI for <100ms p95 inference latency — same pattern would apply to news / market signal serving.
- **Calibration discipline**: Platt scaling reduced ECE 71.5% (0.0312 → 0.0089) on the classification work. In an LLM context, this is the same instinct that says "trust intervals on judge outputs, don't just take the argmax."

I built these because I find the engineering and the statistics genuinely interesting together — feature engineering (47 hand-crafted linguistic/semantic/structural features), Bayesian hierarchical modeling in PyMC, and the production scaffolding all in the same project. That seems like the day-to-day of the role you're hiring for.

MS Applied Statistics, RIT, expected 2026. Reports and code references available. Available for interviews.

Best,
Derek Lankeaux, MS (Applied Statistics, RIT — 2026)
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
