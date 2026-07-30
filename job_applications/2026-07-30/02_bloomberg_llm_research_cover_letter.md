# Cover Letter — Bloomberg, Senior LLM Research Engineer, AI

**Role:** Senior LLM Research Engineer, Artificial Intelligence (Req 17486)
**Location:** New York, NY
**Apply:** https://bloomberg.avature.net/careers/JobDetail/Senior-LLM-Research-Engineer-Artificial-Intelligence/17486
**Lead project:** LLM Ensemble Textbook Bias Detection

*Note: This is a Senior title with Ph.D. or MSc + practical NLP experience. Framing the application as MSc-track candidate with three published technical reports; worth submitting.*

---

Dear Bloomberg AI hiring team,

Bloomberg's LLM group works at the exact intersection I've been building toward: training, tuning, and evaluating LLMs on high-quality domain data, then defending the results with statistical rigor. Two of your named focus areas — **evaluation of LLMs** and **model safety and responsible AI** — describe the last twelve months of my portfolio directly.

The most transferable project is a multi-LLM bias detection system I shipped in April 2026. It processes 4,500 long-form passages (**2.5M tokens, 67,500 ratings**) through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble with async batching, circuit breakers, and exponential backoff on the API layer, and MLflow-tracked runs end-to-end. The eval rubric held **Krippendorff's α = 0.84** and 92% pairwise correlation across the three raters — the reliability floor you need before making publisher-level claims.

Where I'd expect the fit to be strongest for financial NLP work:

- **PyMC Bayesian hierarchical model, partial pooling across publishers**, with MCMC diagnostics that came in cleanly (**R-hat < 1.01, ESS > 1000**) and 95% HDI credible intervals per publisher and per topic. Directional bias was significant at **Friedman χ² = 42.73, p < 0.001** in 3 of 5 publishers, with Nemenyi post-hoc localizing the effects. The same hierarchical structure translates directly to per-issuer / per-sector inference at Bloomberg scale.
- **Multi-testing discipline** — Bonferroni + FDR corrections on 20+ pairwise contrasts, bootstrap CIs on passage-level scores, and a flag for the 12.3% high-uncertainty passages that need human review. That's the pattern for defensible LLM-as-judge deployment in a compliance-sensitive domain.

Two supporting projects: an AI Safety Red-Team framework (96.8% accuracy on 12,500 pairs, 340× cost reduction vs human annotation, α = 0.81) and a clinical-grade ensemble classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987). Together they cover the eval infrastructure, calibration, and SHAP-explainability muscles the JD calls out.

I'm completing an MS in Applied Statistics at RIT (Bayesian inference, MCMC, experimental design). Based in / available for New York, US work authorized, 2026 start. Portfolio and reports at github.com/dl1413.

Best,
Derek Lankeaux
