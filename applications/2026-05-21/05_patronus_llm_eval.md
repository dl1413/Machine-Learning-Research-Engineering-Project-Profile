# Patronus AI — LLM Evaluation Engineer (Remote)

- **Role family:** LLM Eval Platform
- **Lead project:** P1 — AI Safety Red-Team Evaluation (eval harness)
- **Supporting:** P2 — LLM Bias Detection · P3 — Breast Cancer Classification
- **Source:** patronus.ai/careers (Tier B, eval-native)
- **Resume version:** `resume_v3_safety` (Projects section reordered → P1 first)

## JD KEYWORDS TO ECHO (fill from live JD before submit)
`__________`, `__________`, `__________`
Bank: eval harness, LLM-as-judge, autograder, hallucination/harm detection,
production throughput, MLflow tracking, reliability.

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation** — *LLM Evaluation / Eval Infra* bullets
2. LLM Ensemble Bias Detection — *LLM Eval / Research Engineer* bullets
3. Breast Cancer Classification — *Generalist MLE* bullets

## Cover letter

Dear Patronus AI Hiring Team,

I'm interested in Patronus because I've spent the last few months building
exactly the kind of multi-LLM evaluation infrastructure your product abstracts.

I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5, Llama-3.2 —
that auto-grades 12,500 response pairs at 96.8% accuracy and 850 samples/hr,
with circuit breakers, async batching, and MLflow tracking baked in. Cost per
sample landed at $0.018, a 340x reduction versus human review. The interesting
part for Patronus is the stacking layer: a 47-feature meta-classifier that
reconciles disagreement between the three judges and surfaces per-model blind
spots via Bayesian hierarchical modeling with 95% HDI intervals. That maps
directly to the eval-tooling problems you're solving.

My textbook-bias study scaled the same approach — 67,500 ratings over 4,500
passages, circuit-breakered async API integration, MLflow lineage end-to-end,
Krippendorff's alpha = 0.84 — and surfaced publisher bias at p < 0.001, the kind
of result that needs reliability scaffolding to be trusted. A third project, a
clinical-grade classifier at 99.12% accuracy / ROC-AUC 0.9987, shows I treat
latency and explainability (FastAPI <100ms p95, SHAP) as product features.

I'm available for remote, targeting a 2026 start once I wrap my MS in Applied
Statistics at RIT. Portfolio, code, and three technical reports are on my
GitHub (dl1413). I'd love to help turn that eval scaffolding into a product.

Best,
Derek Lankeaux

## Pre-submit checklist
- [ ] 3+ JD phrases echoed in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio links live
- [ ] Work auth: Authorized to work in the US
- [ ] Salary: open, targeting market for role/location
- [ ] JD PDF saved in this folder
