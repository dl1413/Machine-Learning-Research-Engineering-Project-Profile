# Anthropic — Research Engineer, Evaluations

**Date:** 2026-06-05
**Location:** Remote (US) / NYC
**Source:** careers.anthropic.com
**Lead project:** Project 1 — AI Safety Red-Team Evaluation
**Role family:** AI Safety / Eval Engineering
**Resume version:** resume_v3_safety.pdf

---

## JD keywords to mirror (≥3 verbatim in resume + cover letter)
- "model evaluation" / "evaluations"
- "red-teaming"
- "harm" / "harmful behaviors"
- "Constitutional AI"
- "LLM-as-judge"

## Cover letter opener (metric hook)

> My most relevant work for Anthropic's Evaluations team is an independent
> AI Safety Red-Team Evaluation framework I shipped in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains
> a stacking classifier on 47 harm-signal features, reaching **96.8% accuracy
> and ROC-AUC 0.9923** against a 12,500-pair benchmark across 6 harm
> categories. The pipeline runs at 850 samples/hr for $0.018/sample — a
> **340x cost reduction** versus human annotation — while holding
> inter-rater reliability at Krippendorff's alpha = 0.81. I paired it with
> a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals
> per judge, and shipped the whole thing under IEEE 2830-2025 audit-trail
> requirements.

## Resume bullets to surface (top of Projects section)

- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Quantified judge disagreement with PyMC Bayesian hierarchy (95% HDI), surfacing systematic blind spots per model family
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

## Supporting projects (one line each, lower in section)
- LLM Ensemble Textbook Bias Detection — 67,500 ratings, Krippendorff α = 0.84, Friedman χ² = 42.73 (p < 0.001)
- Clinical-Grade Breast Cancer Classifier — 99.12% accuracy, 100% precision, FastAPI <100ms p95

## Submission checklist
- [ ] Resume reordered: Red-Team project on top
- [ ] Cover letter opens with 340x / 96.8% / Krippendorff hook
- [ ] "evaluations", "red-teaming", "harm" appear verbatim
- [ ] Salary expectation: open, targeting market for Research Engineer / remote
- [ ] Work auth: US citizen
