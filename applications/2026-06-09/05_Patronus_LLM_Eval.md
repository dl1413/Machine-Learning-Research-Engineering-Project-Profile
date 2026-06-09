# Patronus AI — LLM Evaluation Engineer

**Location:** Remote (US) · **Lead project:** Project 1 (Red-Team Eval) + Project 2 (Multi-LLM Bias) co-lead
**Source:** patronus.ai/careers · **Snippet base:** APPLICATION_SNIPPETS §1 (Eval Platform) + §2 (LLM Eval)

## JD keyword echo (≥3 verbatim)
- "evaluation harness", "LLM-as-judge", "multi-model", "production scale", "judge disagreement", "eval tooling"

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation** (LEAD — eval harness)
2. **LLM Ensemble Bias Detection** (co-feature — multi-LLM at scale)
3. Breast Cancer ML Classification (production deployment chops)

## Tailored resume bullets
- Built 3-LLM eval harness (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs at **96.8% accuracy and 850 samples/hr**; async batching, circuit breakers, MLflow tracking
- **$0.018/sample** at production scale — **340x cheaper** than human eval — without sacrificing Krippendorff's alpha = 0.81
- 47-feature stacking meta-classifier reconciles **judge disagreement** across model families; PyMC hierarchical Bayes surfaces per-judge blind spots (95% HDI)
- Operated parallel multi-LLM rating pipeline at 67,500 ratings / 2.5M tokens; **Krippendorff's alpha = 0.84**, 92% pairwise correlation across raters
- IEEE 2830-2025 audit trail + model cards baked into every eval run; reproducible from priors through posteriors

## Cover letter

> Dear Patronus AI team,
>
> I'm interested in Patronus because I've spent the last few months
> building exactly the kind of **multi-LLM evaluation harness** your
> product abstracts. My red-team eval pipeline runs GPT-4o, Claude-3.5,
> and Llama-3.2 as judges and auto-grades 12,500 response pairs at
> **96.8% accuracy and 850 samples/hr**, with circuit-breakered async
> batching and MLflow lineage end-to-end. Cost lands at $0.018/sample
> ($6.12 -> $0.018, a **340x reduction** versus human eval). The
> interesting layer for an eval-platform team: a 47-feature stacking
> meta-classifier sits on top of the three judges and a PyMC Bayesian
> hierarchical model produces 95% HDI per-judge blind-spot maps — so
> when judges disagree, the platform can name *why* instead of averaging
> over it.
>
> The same scaffolding ran a bias-detection study at 67,500 ratings and
> 2.5M tokens with **Krippendorff's alpha = 0.84** and surfaced
> publisher-level bias at p < 0.001 (Friedman chi-squared = 42.73). The
> generalizable artifact is the eval harness itself: prompts, rubrics,
> reliability metrics, hierarchical uncertainty model, circuit-breakered
> API layer, audit-grade reporting. That's the thing I'd want to help
> productize.
>
> Finishing my MS in Applied Statistics at RIT (2026). Remote-available
> (Eastern), US-authorized.
>
> Best,
> Derek Lankeaux
