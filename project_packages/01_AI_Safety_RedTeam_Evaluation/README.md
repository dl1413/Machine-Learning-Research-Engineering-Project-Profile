# AI Safety Red-Team Evaluation

**Type:** Independent technical case study
**Focus:** Scalable harm evaluation for LLM responses
**Report version:** 2.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) |
| Publication-formatted report | [PDF](../../AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Manual safety review is costly and difficult to scale. | LLM ensemble annotation followed by supervised classification and Bayesian risk analysis. | Simulated evaluation: 12,500 response pairs; α = 0.81 inter-rater reliability; 96.8% held-out classifier accuracy against ensemble labels. |

## What this demonstrates

- A two-stage evaluation design that separates annotation quality from
  downstream classification performance.
- Reliability analysis, uncertainty quantification, and feature attribution as
  complements to an accuracy metric.
- A structure for audit-oriented safety evaluation and human-review workflows.

## Scope

All data and results are simulated to demonstrate the evaluation design; they
are not measurements of real AI systems. Classifier accuracy is measured
against the LLM ensemble's labels, not human-verified ground truth. This work is not a substitute for expert red-teaming, human safety
review, or a complete deployment-readiness assessment.

## Methods

`LLM evaluation` · `stacking classifier` · `XGBoost` · `PyMC` · `SHAP` · `MLflow`
