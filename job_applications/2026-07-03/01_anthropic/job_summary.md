# Anthropic — Applied AI Engineer / Research Engineer, Evaluations

**Location:** New York, NY (also SF / Seattle / London / Dublin) — remote-eligible for select roles
**Careers page to verify:** https://www.anthropic.com/careers/jobs

## Why this role

Anthropic staffs multiple teams that map directly onto Derek's portfolio:

- **Research Engineer, Evaluations / Safety Evaluations** — Claude eval design, adversarial evals, RSP measurement
- **Applied AI Engineer** — customer-facing LLM eval and pipeline work
- **Frontier Red Team / Model Behavior** — red-teaming, dangerous-capability evals

All three touch the exact stack Derek built independently.

## Key JD requirements (typical, Anthropic Evaluations family)

- Design and run offline evals for frontier models across capabilities and safety
- Build LLM-as-judge / ensemble-of-judges pipelines with measurable reliability
- Statistical rigor on eval results (uncertainty, significance, multiple testing)
- Familiarity with RLHF/RLAIF, Constitutional AI, MITRE ATLAS or equivalent
- Python + production-grade data pipelines; comfort with model APIs at scale

## Anchor projects → JD mapping

| Anthropic need | Portfolio evidence |
|---|---|
| Safety eval design + adversarial taxonomy | AI Safety Red-Team: 8-category MITRE ATLAS-aligned taxonomy, multi-turn escalation identified as highest-risk (31.8%) |
| LLM-as-judge reliability | Krippendorff's α = 0.81 (Red-Team) and 0.84 (Bias) across GPT-4o / Claude-3.5 / Llama-3.2 |
| Cost-efficient scaling | 340× cost reduction, $0.018/sample, 850 samples/hour, 80K+ API calls with circuit breakers |
| Uncertainty on eval results | Bayesian hierarchical modeling, 95% HDI, MCMC R-hat < 1.01, Friedman χ² with correction |
| Production MLOps | MLflow, FastAPI (<100ms p95), SHAP audit trails, IEEE 2830-2025 compliance |

## Compensation & fit

Anthropic technical bands run ~$300K–$425K base (Levels.fyi median TC ~$545K). Derek is early-career (MS 2026), so realistic entry is Research Engineer L3 / Applied AI Engineer L3.

## Application path

1. Search `anthropic.com/careers/jobs` for "Evaluations", "Research Engineer, Safety", "Applied AI Engineer"
2. Prefer NYC-tagged postings first; then SF-remote-eligible
3. Include cover letter + resume + `AI_Safety_RedTeam_Evaluation_Publication.pdf` link
