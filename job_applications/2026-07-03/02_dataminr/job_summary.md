# Dataminr — Senior Research Scientist (NLP, LLM, GenAI)

**Location:** New York, NY (hybrid; some full-remote listings for the same family)
**Careers page to verify:** https://www.dataminr.com/careers and https://www.builtinnyc.com/job/senior-research-scientist/7468278

## Why this role

Dataminr's product is real-time event detection from noisy multilingual public signal — a domain where LLM ensembles + classifier stacks + calibrated probabilities are the core stack. Derek's Red-Team pipeline is architecturally isomorphic: multi-source ingestion → LLM ensemble labeling → classifier with calibrated output → SHAP audit.

## Key JD requirements (typical Dataminr Senior Research Scientist NLP/LLM/GenAI)

- Design and evaluate LLM and hybrid NLP systems for high-throughput classification / clustering
- Production Python; scalable inference; latency SLA discipline
- Statistical validation (precision/recall tradeoffs at threshold, calibration, drift)
- Prompt engineering, retrieval-augmented generation, model-agnostic evaluation
- Communicate results to non-research stakeholders (product, ops, clients)

## Anchor projects → JD mapping

| Dataminr need | Portfolio evidence |
|---|---|
| High-throughput ensemble NLP | Red-Team: 850 samples/hour, 80K+ API calls with circuit breakers |
| Calibrated classification at threshold | Breast Cancer: ECE 0.0089 (71.5% reduction via Platt), context-adaptive thresholds (100% sens at 0.31) |
| Reliability under multi-source disagreement | Bias project: 92% pairwise correlation across GPT-4o/Claude/Llama; Bayesian partial pooling |
| Uncertainty for downstream routing | Bootstrap CIs flag high-uncertainty items for human review (12.3% in Bias project) |
| Production Python + monitoring | MLflow, FastAPI, exponential backoff, drift-ready pipelines |

## Note on Senior title

Dataminr posts "Senior" as their standard IC track — the JD reads Sr-3-to-4 equivalent. Derek should apply and let the recruiter route to the right level; the projects support Sr-3 discussion; new-grad rejection is possible but the cover letter frames MS + three shipped systems as substitute.

## Application path

1. Apply directly via Dataminr careers portal — prefer NYC HQ postings
2. Cross-post via BuiltIn NYC quick-apply
3. Ping LinkedIn recruiters in Dataminr Research org for referral
