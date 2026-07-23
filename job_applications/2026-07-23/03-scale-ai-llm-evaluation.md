# Scale AI — Applied Scientist, LLM Evaluation

**Location:** Remote / San Francisco / NYC · **Apply:** https://scale.com/careers
(filter "LLM Evaluation" or "Applied Scientist") · **Fit:** ★★★★★

---

Dear Scale AI hiring team,

I'm applying for an Applied Scientist role on LLM Evaluation. Cost-to-quality
tradeoffs in labeling pipelines, inter-rater reliability at production scale, and
multi-model ensemble design are the exact problems my published work is about.

My **AI Safety Red-Team Evaluation** project is a working example of the pattern:
GPT-4o + Claude-3.5 + Llama-3.2 as an annotation ensemble feeding a 47-feature
Stacking Classifier, delivering **Krippendorff's α = 0.81** across 12,500 samples
at **$0.018/sample vs $6.12 human baseline (340× cost reduction)** and 850
samples/hour throughput. Precision 97.2%, recall 96.1%, ROC-AUC 0.9923, and a full
MLOps stack with circuit breakers, exponential backoff, MLflow tracking, and SHAP
explanations — the audit-grade evidence a labeling customer would ask for.

The companion project — **LLM Ensemble Bias Detection** — pushes on the measurement
side: 67,500 ratings, 4,500 passages, 2.5M tokens through a production API pipeline,
Bayesian hierarchical model in PyMC with **R-hat < 1.01, 95% HDI, Friedman χ² =
42.73 (p < 0.001)**, Spearman ρ up to 0.74 across publishers, and bootstrap
uncertainty flagging 12.3% of passages for expert review — exactly the human-in-the-
loop pattern Scale's customers rely on. The third project (clinical ML, 99.12%
accuracy, ECE 0.0089 after Platt calibration) rounds out the calibration and
threshold-tuning experience.

I'd like to talk about how these patterns generalize — especially the α-driven
throttling of LLM-vs-human review and the Bayesian uncertainty gating for hard
cases.

Derek Lankeaux · MS Applied Statistics, RIT (2026) ·
[LinkedIn](https://linkedin.com/in/derek-lankeaux) ·
[GitHub](https://github.com/dl1413)
