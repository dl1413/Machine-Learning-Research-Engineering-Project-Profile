# Patronus AI — LLM Evaluation Platform Engineer

**Geo:** Remote (US) / SF + NYC presence
**Role family:** LLM Eval Platform / Eval Infra
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 (multi-LLM eval at scale), P3 (production ML)
**JD link:** <PASTE_JD_URL_AT_SUBMIT>

---

## Resume re-order

1. **AI Safety Red-Team Evaluation Framework** — use "LLM Evaluation / Eval
   Infra roles" bullets (lead with throughput + cost numbers)
2. **LLM Ensemble Textbook Bias Detection** — "LLM Eval / Research
   Engineer" bullets (proves the harness generalizes beyond safety)
3. **Clinical-Grade Breast Cancer ML** — Generalist MLE bullets

## JD phrases to echo (≥3)

- "evaluation platform" / "eval product"
- "LLM-as-judge" / "judge models"
- "production" / "throughput" / "latency"

## Cover letter

> Dear Patronus team,
>
> I'm applying because I've spent the last six months building exactly
> the kind of multi-LLM evaluation infrastructure Patronus is productizing.
> My AI Safety red-team eval is a 3-model harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at **96.8% accuracy
> and 850 samples/hr**, with circuit breakers, exponential backoff,
> async batching, and MLflow run tracking. Cost per sample landed at
> **$0.018, a 340x reduction** versus human review, and a stacking
> meta-classifier on 47 harm-signal features reconciles judge
> disagreement. A PyMC Bayesian hierarchical layer on top surfaces
> per-judge blind spots with 95% HDI risk intervals — the kind of "trust
> the eval result" guarantee your customers actually pay for.
>
> The pattern repeats in my textbook-bias study: 67,500 ratings across
> 4,500 passages on the same 3-LLM ensemble, Krippendorff α = 0.84,
> Friedman χ² = 42.73 (p < 0.001) — same harness, different domain. And
> my clinical breast-cancer classifier (99.12% accuracy, <100ms p95
> FastAPI) shows I can take a model from notebook to a service Patronus
> would actually deploy.
>
> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I finish my MS in Applied Statistics at RIT. Code and
> the three technical reports are on my GitHub (dl1413).
>
> Best,
> Derek Lankeaux
