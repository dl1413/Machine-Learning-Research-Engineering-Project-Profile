# Cover Letter — Traackr, Senior Data Scientist (Remote)

**Job link:** https://jobs.lever.co/traackr/c67bd8c1-6b46-4da3-b136-bb5a6101f54a

---

Dear Traackr Hiring Team,

I'm applying for the Senior Data Scientist role. Your posting stood out because it asks for two things I've spent the last year building: hands-on LLM product work and a practical evaluation discipline — golden sets, regression testing, and human-in-the-loop review. That's exactly the framing I used across my last three projects.

**Practical LLM evaluation that scales.** My AI Safety Red-Team Evaluation framework wraps GPT-4o, Claude-3.5, and Llama-3.2 in a dual-stage ensemble that classifies 12,500 model responses across six harm categories at 96.8% accuracy (Krippendorff's α = 0.81 across judges) and roughly $0.018 per sample — a 340× cost reduction versus the human baseline of $6.12. The pipeline processes 850 samples/hour with circuit breakers, exponential backoff, and MLflow tracking, and every classification is auditable via SHAP. The evaluation harness itself is the reusable part: any team scaling human-in-the-loop labeling can drop into it.

**Statistical rigor behind the recommendation.** In my LLM Ensemble Bias Detection study I processed 67,500 ratings over 4,500 textbook passages (2.5M tokens) and fit a Bayesian hierarchical model in PyMC with partial pooling. R-hat < 1.01, 92% pairwise LLM correlation, Friedman χ² = 42.73 (p < 0.001), and 95% HDIs that surfaced credible bias in 3 of 5 publishers. This is the toolkit I'd bring to Traackr's recommender and experiment work — credible intervals rather than point estimates, uncertainty flags rather than silent failures.

**End-to-end shipping.** My breast-cancer classifier hit 99.12% accuracy and 100% precision with an 8-algorithm benchmark, Platt-calibrated probabilities (ECE 0.0089), and a FastAPI/MLflow deployment at < 100ms p95. The point isn't the medical domain — it's the muscle of taking a modeling question from "framing" to "monitored production" without gaps in calibration, explainability, or documentation.

I'm finishing an MS in Applied Statistics at RIT (Bayesian methods focus) in 2026 and I'm looking for a remote team where LLM evaluation is treated as a first-class engineering problem. Traackr fits that shape. Portfolio, reports, and code are at github.com/dl1413.

Best regards,
Derek Lankeaux
dlankeaux12@gmail.com | linkedin.com/in/derek-lankeaux
