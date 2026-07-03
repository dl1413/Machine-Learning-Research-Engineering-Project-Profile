# Snorkel AI — Applied ML Data Scientist / Applied Research Scientist

**Location:** Remote (US) — Snorkel is remote-first with SF and Redwood City hubs
**Careers page to verify:** https://snorkel.ai/careers/

## Why this role

Snorkel's core thesis is programmatic labeling: labeling functions, weak supervision, LLM-as-judge distilled into deterministic rules. Derek's LLM Ensemble Bias project is essentially "labeling functions" implemented as LLM prompts, with Bayesian aggregation on top — the exact architectural bet Snorkel makes with its LFs. Snorkel's 2026 roadmap ("year of environments") is doubling down on this.

## Key JD requirements (Applied ML Data Scientist)

- Build labeling functions + LLM-as-judge pipelines for enterprise clients
- Weak-supervision modeling; aggregating noisy label sources
- Uncertainty quantification and calibration for label quality
- Prompt engineering + evaluation of LLM annotators
- Python, MLflow-adjacent tracking, client-facing communication

## Anchor projects → JD mapping

| Snorkel need | Portfolio evidence |
|---|---|
| Multiple noisy annotators → clean label | Bias: 3-model ensemble, 92% pairwise correlation, PyMC partial pooling to aggregate |
| LLM-as-judge design + reliability | Both Red-Team (α = 0.81) and Bias (α = 0.84) hit audit-grade reliability |
| Uncertainty-aware label routing | Bootstrap CIs flag 12.3% highest-uncertainty items for human relabel |
| Cost-of-labeling economics | 340× cost reduction on Red-Team; $0.018/sample; 850/hr throughput |
| Cross-source structural analysis | Spearman correlation matrix on Bias project revealed editorial structure (ρ up to 0.74) |

## Application path

1. Snorkel careers portal — filter for "Applied", "Data Scientist", "Research"
2. Snorkel has a research-adjacent applied track — good for MS-with-strong-portfolio profiles
3. Reference labeling-function analogy in cover letter explicitly

## Compensation

Snorkel IC bands: ~$150–200K base + equity, remote-flat.
