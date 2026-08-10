# Resume Tailoring — Anthropic

## Lead order (top-to-bottom)

1. AI Safety Red-Team Evaluation (put first — 100% overlap with Evaluations team)
2. LLM Ensemble Bias Detection (Bayesian rigor + LLM-as-judge reliability)
3. Breast Cancer ML (drop to third — proof of calibration/ECE discipline)

## Keyword swaps

- "harm detection" → "safety evaluation" (Anthropic's term of art)
- "adversarial attack taxonomy" → add "aligned with MITRE ATLAS"
- "Constitutional AI" — already in tech stack, keep visible
- Add "RSP" reference in cover only (not resume — resume stays company-neutral)

## Bullets to strengthen

- Red-Team bullet 1: lead with "Designed and shipped an eval pipeline that produced audit-grade harm labels at $0.018/sample" — cost is the wedge Anthropic cares about
- Bias bullet 2: lead with "Delivered publisher-level credible bias findings (95% HDI) using PyMC partial-pooling with R-hat < 1.01"

## Bullets to soften

- Drop the FastAPI <100ms p95 line — Anthropic serves at their own infra scale, this is undersell

## Publications to attach

- Primary: `AI_Safety_RedTeam_Evaluation_Publication.pdf`
- Secondary link in cover: `LLM_Bias_Detection_Publication.pdf`
