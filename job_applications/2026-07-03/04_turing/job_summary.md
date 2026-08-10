# Turing — Applied Research Scientist, LLM Evaluation & Post-Training

**Location:** Remote (US) — Turing is fully distributed
**Careers page to verify:** https://www.turing.com/careers and https://www.turing.com/jobs/remote-llm-data-scientist

## Why this role

Turing's LLM post-training business runs on rubric-scored eval loops, pass@k metrics, trajectory-level scoring, and RLHF/RLVR data curation for frontier labs. Derek's Bias project is literally a rubric-scored ensemble eval with statistical rigor; the Red-Team project is a rubric-graded safety eval. Turing hires for this profile explicitly.

## Key JD requirements (Applied Research Scientist LLM Eval & Post-Training)

- Define and track eval metrics: pass@k, trajectory scoring, rubric-based scoring
- Feedback signal design for post-training (SFT / RLHF / RLVR / RLAIF)
- Analyze how eval design changes model behavior downstream
- Python, prompt engineering, familiarity with training loop concepts
- Statistical validation of eval results

## Anchor projects → JD mapping

| Turing need | Portfolio evidence |
|---|---|
| Rubric-based scoring at scale | Bias: 67,500 rubric scores across 4,500 items, 3-model ensemble at α = 0.84 |
| Feedback-signal reliability | Red-Team: dual-stage annotation → classifier; per-model risk decomposition |
| Statistical eval analysis | Friedman χ² = 42.73, MCMC R-hat < 1.01, Bootstrap CIs, Bonferroni/FDR |
| Cost-efficient large runs | 340× cost reduction, 850 samples/hr, 80K+ API calls |
| Cross-lens agreement metrics | 92% ensemble pairwise correlation on Bias project |

## Application path

1. Turing careers portal — filter for "Applied Research Scientist" or "LLM Evaluation"
2. Turing also lists client-facing evaluation projects (paid hourly $60+/hr) as a separate track — can dual-apply
3. Include portfolio link — Turing recruiters read it
