# Anthropic — Research Engineer, Model Evaluations

**Geo:** Remote (US) / NYC presence
**Role family:** AI Safety / Eval Engineering
**Lead project:** P1 — AI Safety Red-Team Evaluation Framework
**Supporting:** P2 (Bayesian bias detection rigor), P3 (production ML discipline)
**JD link:** <PASTE_JD_URL_AT_SUBMIT>

---

## Resume re-order (top of Projects section)

1. **AI Safety Red-Team Evaluation Framework** — lead bullets from
   `APPLICATION_SNIPPETS.md` → "AI Safety / Red-Team / Alignment roles"
2. **LLM Ensemble Textbook Bias Detection** — keep T&S bullets
3. **Clinical-Grade Breast Cancer ML** — keep Applied ML bullets

## JD phrases to echo verbatim (≥3)

- "model evaluations" / "evaluation infrastructure"
- "red-teaming" / "adversarial testing"
- "Constitutional AI" (if it appears) — otherwise "policy-violation detection"

## Cover letter

> Dear Anthropic Research Engineering team,
>
> My most relevant work for Anthropic's mission is an independent AI Safety
> Red-Team Evaluation framework I built and published in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains
> a stacking classifier on 47 harm-signal features, reaching **96.8%
> accuracy and ROC-AUC 0.9923** against a 12,500-pair benchmark across 6
> harm categories. The pipeline runs at 850 samples/hr for $0.018/sample —
> a **340x cost reduction** versus human annotation — while holding
> inter-rater reliability at Krippendorff's alpha = 0.81. I paired it with
> a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals
> per judge, and shipped the whole thing under IEEE 2830-2025 audit-trail
> requirements.
>
> Two earlier projects round out the toolkit: a 67,500-rating LLM bias
> study on 4,500 textbook passages (Krippendorff α = 0.84; Friedman χ² =
> 42.73, p < 0.001 across 3/5 publishers), and a clinical breast-cancer
> classifier at 99.12% accuracy / 100% precision deployed behind FastAPI
> at <100ms p95. Together they show I can move from eval design through
> Bayesian uncertainty to production serving without losing rigor at any
> step.
>
> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
> code, and the three technical reports referenced above are on my GitHub
> (dl1413). Happy to walk through any of them — the red-team eval is
> probably the fastest way to see how I think about Anthropic's evaluation
> work.
>
> Best,
> Derek Lankeaux
