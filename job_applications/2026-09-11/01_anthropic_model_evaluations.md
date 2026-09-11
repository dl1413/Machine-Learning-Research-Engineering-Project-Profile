# 01 · Anthropic — Research Engineer, Model Evaluations

- **Company:** Anthropic
- **Role:** Research Engineer, Model Evaluations
- **Location:** San Francisco / New York / Seattle (hybrid) — verify NYC availability on current posting
- **Application URL:** https://job-boards.greenhouse.io/anthropic/jobs/5198255008
- **Referenced comp band (from adjacent role):** ~$320k–$405k base
- **Priority:** ⭐⭐⭐ (best single fit)

## Why this fits Derek

Model Evaluations at Anthropic sits at the exact intersection of two of the
three primary projects: (a) the AI Safety Red-Team Evaluation project, which
built a two-stage LLM-ensemble + supervised classifier over 12,500 response
pairs with α = 0.81 and 96.8% held-out accuracy, and (b) the LLM Ensemble
Textbook Bias Detection project, which produced 67,500 rubric-based ratings
across 4,500 passages with α = 0.84 and Bayesian partial-pooling for
publisher-level effects. Both projects operationalize the eval-design
practice the team owns: separating annotation reliability from downstream
classifier performance, quantifying inter-rater agreement rigorously, and
carrying uncertainty through to the final review decision.

## Cover letter (draft)

Dear Anthropic Model Evaluations team,

I'm applying for the Research Engineer, Model Evaluations role because the
two evaluation projects I've built this year map directly onto what your team
does day-to-day. In the AI Safety Red-Team Evaluation study, I designed a
two-stage workflow that combined an LLM-judge ensemble with a supervised
harm classifier over 12,500 response pairs across six harm categories, and
reported Krippendorff's α = 0.81 alongside 96.8% held-out classification
accuracy — deliberately keeping annotation reliability, model performance,
and uncertainty as distinct measurements rather than collapsing them into
one headline number. In the LLM Ensemble Textbook Bias Detection project I
scaled the same pattern to 67,500 rubric-based ratings across 4,500 passages,
reached α = 0.84, and used PyMC partial pooling to disentangle
publisher-level from passage-level effects with proper MCMC diagnostics.

What I want to build next is exactly the harness-level work your posting
describes: reusable eval infrastructure that catches regressions before they
ship, and measurement designs that hold up when a model change moves a
metric for reasons the metric wasn't built to detect. My statistics
background (M.S., Applied Statistics, RIT) is the reason I default to
reliability-then-accuracy in that order.

I'd be glad to talk about the eval-design choices I'd revisit today. Full
technical reports for both projects, with methods and limitations, are in
the portfolio linked below.

— Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Tailored resume bullets

- Designed and ran a two-stage LLM-ensemble + supervised harm-classification
  evaluation over 12,500 response pairs across six harm categories;
  Krippendorff's α = 0.81, 96.8% held-out accuracy, uncertainty reported
  separately from point estimates.
- Scaled rubric-based LLM-as-judge evaluation to 67,500 ratings across 4,500
  textbook passages; reported α = 0.84 and modeled publisher-level effects
  with PyMC partial pooling and MCMC diagnostics.
- Built evaluation harness patterns that keep annotation reliability,
  classifier performance, and uncertainty as separate first-class metrics.

## Follow-ups

- If a NYC posting appears under the same job family, apply to both (they
  route to the same team).
- Include the AI Safety and LLM Bias PDFs from `project_packages/` in the
  application's "portfolio / additional materials" field if available.
