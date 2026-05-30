# Anthropic — Research Engineer, Evaluations

**Location:** Remote / NYC / SF
**Role family:** AI Safety / LLM Evaluation
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting:** LLM Ensemble Bias Detection, Breast Cancer Classification (rigor)
**JD link:** https://www.anthropic.com/jobs (paste exact posting URL on submit)
**Resume version:** Resume_Derek_Lankeaux_v_safety.pdf
**Status:** ready_to_submit

---

## Cover Letter

Dear Anthropic Hiring Team,

I'm applying for the Research Engineer, Evaluations role because the work I've spent the last several months on lines up almost one-for-one with what your evals team ships. I built an independent AI Safety Red-Team Evaluation framework that ensembles GPT-4o, Claude-3.5-Sonnet, and Llama-3.2 as red-team judges, then trains a stacking meta-classifier on 47 harm-signal features over a 12,500-pair benchmark across 6 harm categories. The pipeline reaches **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**, runs at **850 samples/hour for $0.018/sample — a 340x cost reduction versus human annotation** — while holding inter-rater reliability at **Krippendorff's alpha = 0.81**. A PyMC Bayesian hierarchical layer produces 95% HDI risk intervals per judge so disagreement becomes a measurable signal rather than noise. The whole thing ships under IEEE 2830-2025 audit-trail requirements with SHAP explanations on every prediction.

Two supporting projects extend the same skillset: a multi-LLM textbook bias study (67,500 ratings, 2.5M tokens, alpha = 0.84, Friedman chi-squared = 42.73, p < 0.001) and a clinical-grade classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987) — together they show I treat eval rigor and production reliability as the same problem.

I'd love to bring that combination of multi-judge eval design, statistical defensibility, and production throughput to Anthropic's evaluations work — especially anything that touches Constitutional AI, jailbreak resistance, or model-behavior measurement.

Available for NYC on-site or remote, targeting a 2026 start once I wrap my MS in Applied Statistics at RIT. Code and full technical report on GitHub (dl1413).

Best,
Derek Lankeaux

---

## Top Resume Bullets (paste at top of Projects section)

- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 to $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across three LLM judges
- Built production eval harness sustaining 850 samples/hr with circuit breakers, exponential backoff, async batching, and MLflow lineage
- Quantified judge disagreement via PyMC Bayesian hierarchical model with 95% HDI risk intervals; shipped under IEEE 2830-2025 audit-trail requirements

## JD Keywords to Echo
LLM evaluation, red-teaming, jailbreak, Constitutional AI, model behavior, harm classification, inter-rater reliability, eval harness, SHAP, responsible AI.
