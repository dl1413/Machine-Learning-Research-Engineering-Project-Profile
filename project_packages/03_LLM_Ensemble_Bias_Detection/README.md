# LLM Ensemble Textbook Bias Detection

**Type:** Independent technical case study
**Focus:** Uncertainty-aware LLM evaluation for educational-content review
**Report version:** 4.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../LLM_Ensemble_Bias_Detection_Report.md) |
| Publication-formatted report | [PDF](./LLM_Bias_Detection_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Evaluate whether multiple LLM judges can support large-scale content review. | Rubric-based rating, inter-rater reliability, Bayesian hierarchical modeling, and uncertainty triage. | 4,500 passages; 67,500 ratings; α = 0.84; MCMC R-hat < 1.01. |

## What this demonstrates

- How to measure agreement before aggregating LLM judgments.
- How partial pooling and posterior intervals can make uncertainty visible in
  group-level comparisons.
- A review workflow that routes disagreement and high-uncertainty material for
  expert inspection.

## Scope

The analysis is a research framework based on the report's corpus, rubric, and
model prompts. Its findings require expert human review and should not be read
as a general claim about publishers, textbooks, or political bias outside that
study design.

## Methods

`LLM-as-judge` · `Krippendorff's alpha` · `PyMC` · `ArviZ` · `FastAPI` · `MLflow`
