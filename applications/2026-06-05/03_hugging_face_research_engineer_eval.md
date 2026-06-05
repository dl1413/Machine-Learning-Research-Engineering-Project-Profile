# Hugging Face — Research Engineer, Evaluation

**Date:** 2026-06-05
**Location:** Remote / NYC
**Source:** huggingface.co/jobs
**Lead project:** Project 2 — LLM Ensemble Bias Detection
**Role family:** LLM Evaluation / Trust & Safety
**Resume version:** resume_v3_eval.pdf

---

## JD keywords to mirror
- "open evaluation" / "open-source eval"
- "bias" / "fairness"
- "multi-model"
- "reproducibility"
- "leaderboard" / "benchmark"

## Cover letter opener (metric hook)

> I'm interested in Hugging Face because I've spent the last few months
> building exactly the kind of multi-LLM evaluation infrastructure your
> Open LLM Leaderboard and Evaluate library exist to standardize. My
> textbook-bias study ran **67,500 ratings through a GPT-4o / Claude-3.5
> / Llama-3.2 ensemble over 4,500 passages (2.5M tokens)**, with
> circuit-breakered async API integration and MLflow lineage end-to-end.
> The eval rubric held **Krippendorff's alpha = 0.84** and surfaced
> publisher-level bias at **p < 0.001** — the kind of result that needs
> the reliability scaffolding to be trusted. I'd love to help turn that
> scaffolding into open infrastructure at HF.

## Resume bullets to surface

- Designed multi-LLM bias-rating system covering 4,500 textbook passages and 2.5M tokens; surfaced significant publisher-level bias (p < 0.001) in 3/5 publishers
- Held inter-rater reliability at Krippendorff's alpha = 0.84 and 92% pairwise correlation across GPT-4o, Claude-3.5, Llama-3.2
- Built async API layer with circuit breakers and exponential backoff sustaining 67,500 rated samples without manual intervention
- Published reproducible technical report with methodology, priors, sensitivity analysis, and full posterior visualizations

## Supporting projects
- AI Safety Red-Team Eval — 96.8% accuracy, 12,500 pairs, IEEE 2830-2025 audit trail
- Breast Cancer Classifier — 99.12% accuracy, productionized

## Submission checklist
- [ ] Resume reordered: Bias Detection + Red-Team on top
- [ ] Cover letter opens with 67,500 / alpha = 0.84 / p < 0.001 hook
- [ ] "bias", "multi-model", "reproducibility" verbatim
- [ ] GitHub URL prominent (open-source orientation)
- [ ] Salary expectation: open
