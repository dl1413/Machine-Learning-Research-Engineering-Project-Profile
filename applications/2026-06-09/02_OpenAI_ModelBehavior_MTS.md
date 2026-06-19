# OpenAI — Member of Technical Staff, Model Behavior & Evals

**Location:** New York, NY · **Lead project:** Project 1 (Red-Team) + Project 2 (Bias) co-lead
**Source:** openai.com/careers · **Snippet base:** APPLICATION_SNIPPETS §1 (LLM Eval Infra) + §2 (Bayesian)

## JD keyword echo (≥3 verbatim)
- "model behavior", "evaluation harness", "LLM-as-judge", "inter-rater reliability", "post-training", "safety evals"

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation Framework** (LEAD)
2. **LLM Ensemble Textbook Bias Detection** (co-feature)
3. Breast Cancer ML Classification

## Tailored resume bullets
- Built production eval harness processing **850 samples/hr** with circuit breakers, exponential backoff, async batching, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 LLM-as-judge into XGBoost meta-classifier reaching **96.8% agreement** with gold human labels on 12,500 pairs
- Modeled judge disagreement via PyMC Bayesian hierarchy (R-hat < 1.01, ESS > 1000) producing 95% HDI per-model blind-spot maps
- Operated parallel 3-LLM bias-rating pipeline at 67,500 ratings / 2.5M tokens; **Krippendorff's alpha = 0.84**, 92% pairwise correlation
- Detected publisher-level bias at Friedman chi-squared = 42.73, **p < 0.001**, with Nemenyi post-hoc localization

## Cover letter

> Dear OpenAI Model Behavior team,
>
> I'm applying because the last six months of my work map almost exactly to
> what your team owns. I built a 3-model **LLM-as-judge** eval harness
> (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) that auto-grades 12,500 response
> pairs at **96.8% accuracy** and 850 samples/hr — circuit-breakered async
> batching, MLflow lineage, $0.018/sample, **340x cheaper** than the human
> baseline. The interesting layer is reconciliation: a 47-feature stacking
> meta-classifier on top of the three judges, plus a PyMC Bayesian
> hierarchical model that emits 95% HDI intervals per judge family so we
> can name systematic blind spots instead of waving at disagreement.
>
> I ran the same multi-LLM stack on a separate problem — textbook bias
> rating across 4,500 passages and **67,500 ratings (2.5M tokens)** — and
> held **inter-rater reliability at Krippendorff's alpha = 0.84** while
> surfacing significant publisher-level bias (Friedman chi-squared = 42.73,
> p < 0.001). Same scaffolding, different domain; both shipped as
> publication-grade technical reports with model cards and reproducibility
> artifacts aligned with IEEE 2830-2025.
>
> Wrapping my MS in Applied Statistics (RIT, 2026). NYC-available,
> remote-friendly, work-authorized. Happy to walk through either pipeline
> on a call.
>
> Best,
> Derek Lankeaux
