# Cover Letter Template — Applied / Production ML Engineer

**Best fit for:** NYC product-ML teams (Spotify, Etsy, Bloomberg, Two Sigma, Datadog, Ramp, Hugging Face NYC, Squarespace, Peloton, Vimeo), Series B–D startups with shipped ML products, fintech/adtech/healthtech with regulated ML.

**Anchor project:** Breast Cancer Classification (lead heavily — it's the cleanest production-ML story). Secondary: AI Safety Red-Team for ensemble depth.

---

Dear {{Hiring Manager Name | Hiring Team}},

I'm applying for the **{{Role Title}}** position at **{{Company}}**. {{One sentence about a specific product, model, or engineering blog post — concrete reference}}.

I'm a Machine Learning Research Engineer (MS Applied Statistics, RIT, 2026) whose work has focused on the parts of ML that actually have to survive production: ensemble systems with calibrated uncertainty, latency-bound serving, and monitoring that catches drift before users do.

A few things I've shipped:

- **Clinical-Grade Breast Cancer Classifier** — 99.12% accuracy with **100% precision** (zero false positives) and ROC-AUC 0.9987, deployed via FastAPI at **<100ms p95 latency**. Built on an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting), Optuna TPE hyperparameter search (5× faster convergence than grid), Platt calibration (ECE 0.0312 → 0.0089, a 71.5% reduction), and SHAP explainability. Full MLflow registry + model cards.

- **LLM Annotation Pipeline at Scale** — Processed **80,000+ API calls** across GPT-4o, Claude-3.5, and Llama-3.2 with circuit breakers, exponential backoff, adaptive rate limiting, and MLflow experiment tracking — sustained 850 samples/hour with no manual intervention.

- **AI Safety Red-Team Evaluation** — Dual-stage LLM ensemble + Stacking Classifier achieving 96.8% accuracy on harm detection at $0.018/sample (340× cheaper than human raters), with Krippendorff's α = 0.81 reliability and SHAP audit trails.

Three things I'd bring to {{Company}}:

1. **Production hygiene as a default** — Circuit breakers, calibration, drift monitoring, A/B scaffolding, and SHAP for stakeholder-facing explainability aren't add-ons in my work; they're in the v1.
2. **Calibrated confidence, not just accuracy** — In regulated or user-facing contexts, the probability matters as much as the prediction. I measure ECE and pick thresholds against operating-point constraints.
3. **Ensembles I can actually serve** — I've benchmarked 8+ algorithms head-to-head and picked the one that meets both accuracy and latency targets, not the one with the prettiest paper.

I'd be glad to walk through any of this in more depth. Portfolio: [dl1413.github.io/LLM-Portfolio](https://dl1413.github.io/LLM-Portfolio/) · GitHub: [github.com/dl1413](https://github.com/dl1413).

Best,

Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · US work auth · Remote / NYC

---

### Customization notes
- For NYC fintech (Bloomberg, Two Sigma, Ramp, Capital One ML): emphasize calibration, regulatory framing (IEEE 2830-2025), and lead with the Breast Cancer + 80K-API-call story.
- For consumer/product (Spotify, Etsy, Peloton): de-emphasize compliance, replace point 3 with "Velocity from notebook to FastAPI endpoint."
- For startups: cut the third bullet, shorten to ~250 words.
