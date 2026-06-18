# 01 — Anthropic · Research Engineer, Model Evaluations

- **Date pulled:** 2026-06-15
- **Location:** New York, NY (Anthropic is office-first; NYC hub)
- **Source:** LinkedIn / Anthropic Greenhouse board
- **JD link:** https://job-boards.greenhouse.io/anthropic/jobs/5198255008
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (LLM Bias Detection — multi-judge reliability), Project 3 (rigor)
- **Resume version to send:** `resume_v3_safety.pdf`
- **Cover letter:** Yes (frontier lab — always include)

## JD signals (from public summary)

> "Design and implement evaluations across the full spectrum of Claude's
> capabilities and personality… build the infrastructure that runs them
> reliably at scale, partnering closely with researchers throughout the
> lifecycle of a new capability."

Keywords to echo verbatim: `evaluations`, `evaluation infrastructure`,
`reliably at scale`, `Claude`, `capabilities`, `researchers`.

## Tailored cover-letter opener (paste-ready)

> Hi Anthropic team — I'd like to be considered for the Research Engineer,
> Model Evaluations role in New York. My most relevant work is an AI Safety
> Red-Team Evaluation framework I built and published in April 2026: it
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains
> a stacking classifier on 47 harm-signal features, reaching **96.8%
> accuracy and ROC-AUC 0.9923** on a 12,500-pair benchmark across 6 harm
> categories. The pipeline runs at **850 samples/hr for $0.018/sample — a
> 340× cost reduction versus human annotation** — while holding inter-rater
> reliability at Krippendorff's α = 0.81. I paired it with a PyMC Bayesian
> hierarchical model that produces 95% HDI risk intervals per judge, all
> shipped under IEEE 2830-2025 audit-trail requirements. That combination
> of eval rigor and production throughput is what I'd want to bring to
> Claude's evaluation infrastructure.

## Resume bullets to surface in top of Projects section

(from `APPLICATION_SNIPPETS.md` → Project 1 → ML Research Engineer)

- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Modeled multi-judge uncertainty via PyMC Bayesian hierarchy, producing 95% HDI risk intervals for downstream policy decisions

## Application checklist

- [x] Lead project surfaced first
- [x] 3+ JD phrases echoed (`evaluations`, `evaluation infrastructure`, `reliably at scale`)
- [x] Metric hook in opener (340×, 96.8%, α = 0.81)
- [x] Portfolio + GitHub + LinkedIn URLs current
- [x] Work-auth: US authorized
- [x] Salary expectation: open, targeting NYC market for the role
