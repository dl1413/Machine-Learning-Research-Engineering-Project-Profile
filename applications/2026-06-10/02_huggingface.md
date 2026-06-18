# Hugging Face — ML Research Engineer, Evaluation

**Geo:** Remote (US) / NYC
**Role family:** LLM Evaluation / Eval Infra
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 (multi-LLM eval at scale), P3 (production discipline)
**JD link:** <PASTE_JD_URL_AT_SUBMIT>

---

## Resume re-order

1. **AI Safety Red-Team Evaluation Framework** — use "LLM Evaluation / Eval
   Infra roles" bullets
2. **LLM Ensemble Textbook Bias Detection** — use "LLM Eval / Research
   Engineer" bullets
3. **Clinical-Grade Breast Cancer ML** — Generalist MLE bullets

## JD phrases to echo (≥3)

- "evaluation harness" / "eval framework"
- "LLM-as-judge" / "model behavior"
- "open-source" / "reproducible" (HF cultural keywords — both apply)

## Cover letter

> Dear Hugging Face team,
>
> I'm applying because the evaluation tooling I've been building solo for
> the last six months is exactly the surface area your team owns. My
> AI Safety Red-Team eval is a 3-model harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at **96.8% accuracy
> and 850 samples/hr**, with circuit breakers, async batching, and MLflow
> tracking baked in. Cost per sample landed at $0.018, a 340x reduction
> versus human review. The interesting part for Hugging Face is the
> stacking layer: a 47-feature meta-classifier that reconciles
> disagreement between the three judges and surfaces per-model blind
> spots via Bayesian hierarchical modeling. That maps directly to the
> `evaluate` / `lighteval` direction you've been pushing.
>
> Two adjacent projects: a 67,500-rating LLM bias study (Krippendorff
> α = 0.84; p < 0.001 publisher effects via Friedman + Nemenyi), and a
> clinical-grade classifier at 99.12% accuracy deployed behind FastAPI.
> All three are open and reproducible on my GitHub (dl1413) — code, full
> technical reports, and PDFs.
>
> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I finish my MS in Applied Statistics at RIT. The
> red-team eval is probably the fastest way to see how I think about
> Hugging Face's evaluation surface.
>
> Best,
> Derek Lankeaux
