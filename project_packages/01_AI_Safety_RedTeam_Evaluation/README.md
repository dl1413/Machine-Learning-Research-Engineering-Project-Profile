# AI Safety Red-Team Evaluation

**Type:** Independent technical case study
**Focus:** Scalable harm evaluation for LLM responses
**Report version:** 2.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) |
| Publication-formatted report | [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Manual safety review is costly and difficult to scale. | LLM ensemble annotation followed by supervised classification and Bayesian risk analysis. | 12,500 response pairs; α = 0.81 inter-rater reliability; 96.8% held-out classifier accuracy. |

## What this demonstrates

- A two-stage evaluation design that separates annotation quality from
  downstream classification performance.
- Reliability analysis, uncertainty quantification, and feature attribution as
  complements to an accuracy metric.
- A structure for audit-oriented safety evaluation and human-review workflows.

## Scope

The reported results apply to the project's experimental setup and annotation
rubric. This work is not a substitute for expert red-teaming, human safety
review, or a complete deployment-readiness assessment.

## Methods

`LLM evaluation` · `stacking classifier` · `XGBoost` · `PyMC` · `SHAP` · `MLflow`
