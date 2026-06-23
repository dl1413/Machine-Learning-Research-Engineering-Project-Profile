# Application 1 — Anthropic · Research Engineer, Model Evaluations

| Field | Value |
|---|---|
| Date | 2026-05-25 (Mon) |
| Company | Anthropic |
| Role | Research Engineer, Model Evaluations |
| Role family | AI Safety / Eval Engineering |
| Location | Remote (US) / NYC office |
| Source | careers.anthropic.com |
| Lead project | Project 1 — AI Safety Red-Team Evaluation Framework |
| Supporting | Project 2 — LLM Bias Detection (rater reliability), Project 3 — Breast Cancer (rigor) |
| Resume version | resume_v3_safety.pdf |
| Cover letter | Yes |
| Referral | No |

---

## Tailored resume bullets (top of Projects section)

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 to $0.018/sample) while holding Krippendorff's alpha = 0.81 across the three judges
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Modeled judge disagreement with a PyMC Bayesian hierarchy (95% HDI), surfacing systematic blind spots per model family
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability, MLflow lineage, and 850 samples/hr throughput

## Cover letter

Dear Anthropic Hiring Team,

My most relevant work for Anthropic's Model Evaluations team is an independent
AI Safety Red-Team Evaluation framework I built and published in January 2026.
It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
stacking classifier on 47 harm-signal features, reaching **96.8% accuracy and
ROC-AUC 0.9923** against a 12,500-pair benchmark across 6 harm categories.
The pipeline runs at **850 samples/hr for $0.018/sample — a 340x cost
reduction** versus human annotation — while holding inter-rater reliability at
**Krippendorff's alpha = 0.81**. I paired it with a PyMC Bayesian hierarchical
model that produces 95% HDI risk intervals per judge, and shipped the whole
thing under IEEE 2830-2025 audit-trail requirements.

Two adjacent projects round out the picture: a multi-LLM bias-rating study
(67,500 ratings, alpha = 0.84, Friedman p < 0.001 on publisher effects) and a
clinical-grade classifier (99.12% accuracy, 100% precision, <100ms p95
FastAPI). Both reflect the same habit Anthropic's eval work depends on —
building the reliability scaffolding before trusting the result.

I'm finishing my MS in Applied Statistics at RIT (2026), based in / available
for NYC, and open to remote. I'd love to bring eval rigor and production
throughput to the Model Evaluations team.

Best,
Derek Lankeaux

## JD keyword echoes to include verbatim

- "model evaluations"
- "red-teaming"
- "LLM-as-judge"
- "Constitutional AI" (skills line)
- "inter-rater reliability"

## Quality checklist

- [x] Lead project at top of Projects section (Red-Team)
- [x] Metric hook opener (340x cost reduction)
- [x] 3+ JD phrases verbatim in resume + cover letter
- [x] LinkedIn / GitHub / Portfolio links current
- [x] Anthropic-friendly framework signal: Claude-3.5 in stack, Constitutional AI in skills
- [x] Salary: "open, targeting market for the role/location"
- [x] Work auth: US authorized
