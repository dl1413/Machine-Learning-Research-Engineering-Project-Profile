# Scale AI — LLM Evaluation Engineer

**Location:** New York, NY (hybrid) / Remote-friendly
**Role family:** LLM Evaluation / Eval Infra
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting:** LLM Ensemble Bias Detection (multi-LLM ensemble at scale)
**JD link:** https://scale.com/careers (paste exact posting URL on submit)
**Resume version:** Resume_Derek_Lankeaux_v_eval.pdf
**Status:** ready_to_submit

---

## Cover Letter

Dear Scale AI Hiring Team,

I'm interested in the LLM Evaluation Engineer role because Scale is solving — as a product — the exact problem I've spent the last several months solving as an independent research project. I built a 3-model LLM eval harness (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) that auto-grades 12,500 response pairs across 6 harm categories at **96.8% accuracy, 850 samples/hour, and $0.018/sample — a 340x reduction versus human annotation** — with circuit breakers, async batching, exponential backoff, and MLflow lineage end-to-end. The stacking meta-classifier reconciles disagreement between the three judges via 47 engineered features and a PyMC Bayesian hierarchical layer surfacing per-model blind spots with 95% HDI intervals. Inter-rater reliability lands at **Krippendorff's alpha = 0.81**.

To show that this isn't a one-off: I ran the same ensemble pattern at larger scale on a textbook-bias study — 67,500 LLM ratings over 4,500 passages and 2.5M tokens, Krippendorff's alpha = 0.84, 92% pairwise correlation, and Friedman chi-squared = 42.73 (p < 0.001) localizing publisher-level bias. Both projects were built so the result could be defended in front of a skeptical reviewer, not just a metric dashboard.

That combination of multi-LLM ensembling, production throughput, and statistical rigor is what I'd want to bring to Scale's eval platform — particularly anything around LLM-as-judge reliability, human-in-the-loop quality, or RLHF data integrity.

Based in / available for NYC, open to remote. Targeting a 2026 start after I finish my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux

---

## Top Resume Bullets

- Built production LLM eval harness processing 850 samples/hr with circuit breakers, exponential backoff, async API integration, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into XGBoost meta-classifier; 96.8% agreement with gold human labels on 12,500 pairs, ROC-AUC 0.9923
- Operated 3-LLM ensemble at production scale: 67,500 ratings, 2.5M tokens, Krippendorff's alpha = 0.84, 92% pairwise correlation
- Quantified per-judge blind spots with PyMC Bayesian hierarchy producing 95% HDI intervals — disagreement becomes a measurable signal

## JD Keywords to Echo
LLM-as-judge, eval harness, model behavior, inter-rater reliability, Krippendorff's alpha, RLHF data quality, prompt iteration, regression testing, async API, MLflow.
