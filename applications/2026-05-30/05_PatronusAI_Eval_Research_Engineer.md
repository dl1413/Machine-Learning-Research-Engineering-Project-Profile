# Patronus AI — Eval Research Engineer

**Location:** Remote (US) / NYC-friendly
**Role family:** LLM Evaluation Platform
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting:** LLM Ensemble Bias Detection (multi-LLM at scale)
**JD link:** https://www.patronus.ai/careers (paste exact posting URL on submit)
**Resume version:** Resume_Derek_Lankeaux_v_eval.pdf
**Status:** ready_to_submit

---

## Cover Letter

Dear Patronus AI Hiring Team,

I'm reaching out about the Eval Research Engineer role because I've spent the last several months building — as an independent research project — exactly the kind of multi-LLM evaluation infrastructure your platform productizes.

The core artifact is a 3-model LLM eval harness (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) auto-grading **12,500 response pairs across 6 harm categories at 96.8% accuracy and 850 samples/hour**, with circuit breakers, async batching, exponential backoff, and MLflow tracking baked in. The interesting layer for Patronus is the stacking meta-classifier: 47 engineered features over the three judges' outputs, surfacing per-model blind spots that no single LLM-judge catches, and a PyMC Bayesian hierarchical model producing 95% HDI risk intervals per judge — disagreement quantified rather than averaged away. Cost landed at **$0.018/sample (340x reduction vs human annotation)** while inter-rater reliability held at **Krippendorff's alpha = 0.81**.

To show the pattern generalizes: I ran a larger-scale bias-detection ensemble — **67,500 LLM ratings, 4,500 passages, 2.5M tokens**, alpha = 0.84, 92% pairwise correlation, Friedman chi-squared = 42.73 (p < 0.001) localizing publisher-level bias. Same async-API scaffolding, same MLflow lineage, same statistical defensibility — which is the part that matters for a product whose value proposition is "trust this eval."

That mix of multi-judge ensembling, eval-rubric design, throughput engineering, and statistical defensibility is what I'd want to bring to Patronus's platform — particularly anywhere LLM-as-judge reliability, hallucination eval, or regression-testing at scale needs harder guarantees.

Remote-friendly, also open to NYC. Targeting a 2026 start after my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux

---

## Top Resume Bullets

- Built production LLM eval harness processing 850 samples/hr with circuit breakers, exponential backoff, async API integration, and MLflow lineage
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into XGBoost meta-classifier; 96.8% agreement with gold human labels at $0.018/sample (340x cost reduction)
- Engineered 47 linguistic / semantic / structural harm features capturing jailbreak, refusal-evasion, and policy-violation signals
- Quantified judge disagreement via PyMC Bayesian hierarchical model with 95% HDI risk intervals; published reproducible technical report

## JD Keywords to Echo
LLM-as-judge, eval platform, hallucination detection, regression testing, prompt evaluation, multi-model ensembling, async API, circuit breakers, MLflow, reproducible benchmarks.
