# Resume bullets — Patronus AI, Machine Learning Engineer (Remote)

Lead with Project 1 (Eval Platform / LLM Evaluation version). Supporting: 2 + 3.

## AI Safety Red-Team Evaluation Framework (LEAD)

- Shipped 3-LLM eval harness (GPT-4o, Claude-3.5, Llama-3.2) auto-grading **12,500 response pairs at 96.8% accuracy, ROC-AUC 0.9923**
- Production throughput: **850 samples/hr at $0.018/sample (340x cost reduction vs human review)**
- Stacking meta-classifier over 47 harm-signal features reconciles inter-judge disagreement; Krippendorff's alpha = 0.81
- PyMC Bayesian hierarchical model surfaces per-judge blind spots with 95% HDI intervals — calibrated probabilities, not raw scores
- Eval harness ships with circuit breakers, exponential backoff, async batching, MLflow lineage end-to-end

## LLM Ensemble Bias Detection (Supporting — eval rubric @ scale)

- 67,500 LLM ratings over **4,500 passages, 2.5M tokens**; rubric held Krippendorff's alpha = 0.84
- 92% pairwise inter-LLM correlation across GPT-4o / Claude-3.5 / Llama-3.2
- Friedman test (chi-squared = 42.73, p < 0.001) + PyMC partial pooling (R-hat < 1.01) surfaced publisher-level bias with credible intervals

## Clinical-Grade Classifier (Supporting — production ML rigor)

- 8-algorithm benchmark under nested CV; winning ensemble at **99.12% accuracy, 100% precision, ROC-AUC 0.9987**
- FastAPI service with MLflow registry; <100ms p95 latency

## JD keyword echo
LLM evaluation, eval harness, LLM-as-judge, adversarial test cases,
hallucination detection, PII detection, benchmark, jailbreak, red-team,
content safety, eval platform, MLflow, async, circuit breaker, ensemble
judge, calibration, Bayesian, uncertainty quantification.
