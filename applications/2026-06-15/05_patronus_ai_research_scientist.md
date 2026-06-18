# 05 — Patronus AI · Research Scientist (Remote)

- **Date pulled:** 2026-06-15
- **Location:** Remote (US)
- **Source:** Glassdoor / patronus.ai/join
- **JD link:** https://patronus.ai/join
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 — LLM Ensemble Bias Detection
- **Resume version to send:** `resume_v3_eval_platform.pdf`
- **Cover letter:** Yes

## JD signals (from public site)

> "Patronus AI is the leading automated AI evaluation and security
> company. Research Scientists solve important research problems
> surrounding AI evaluation, language model understanding, and robustness
> challenges."

Keywords to echo: `automated evaluation`, `LLM evaluation`, `robustness`,
`language model understanding`, `eval harness`, `LLM-as-judge`,
`hallucination`, `factuality`.

## Tailored cover-letter opener (paste-ready)

> I'm interested in Patronus because I've spent the last few months
> building exactly the kind of multi-LLM evaluation infrastructure your
> product abstracts. My AI Safety Red-Team Evaluation ships a 3-model LLM
> eval harness — GPT-4o, Claude-3.5, Llama-3.2 — that auto-grades 12,500
> response pairs at **96.8% accuracy and 850 samples/hr**, with circuit
> breakers, async batching, and MLflow tracking baked in. Cost per sample
> landed at **$0.018, a 340× reduction versus human review**. The
> interesting part for Patronus is the stacking layer: a 47-feature
> meta-classifier that reconciles disagreement between the three judges
> and surfaces per-model blind spots via Bayesian hierarchical modeling
> (95% HDI per judge). I followed it up with a 67,500-rating bias-
> detection study (Krippendorff's α = 0.84, Friedman χ² = 42.73,
> p < 0.001) using the same infrastructure pattern. That maps directly to
> the eval-tooling problems Patronus is solving as a product.

## Resume bullets to surface

(from `APPLICATION_SNIPPETS.md` → Project 1 → LLM Evaluation)

- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family
- Operated 3-LLM ensemble at production scale across a second study: 67,500 ratings, 2.5M tokens, full MLflow lineage

## Application checklist

- [x] Lead project surfaced first
- [x] 3+ JD phrases echoed (`automated evaluation`, `LLM evaluation`, `robustness`)
- [x] Metric hook in opener (96.8%, 340×, α = 0.81/0.84)
- [x] Remote-friendly (Patronus is remote-first per careers page)
- [x] Work-auth: US authorized
- [x] Salary expectation: open
