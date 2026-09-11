# 05 · Anthropic — Research Engineer, AI Observability

- **Company:** Anthropic
- **Role:** Research Engineer, AI Observability
- **Location:** San Francisco (primary listing); Anthropic also hires
  NYC / Seattle for this team — check current posting for NYC eligibility
- **Application URL:** https://job-boards.greenhouse.io/anthropic/jobs/5125083008
- **Referenced comp band:** ~$320k–$405k base (per search snippet)
- **Priority:** ⭐⭐ (companion Anthropic application — different team from #1)

## Why this fits Derek

AI Observability at Anthropic uses Claude to make sense of the exploding
data volume that comes with production model deployments. Two of Derek's
primary projects are the same shape: the AI Safety Red-Team Evaluation
uses an LLM ensemble to structure high-volume unstructured judgments about
model behavior (α = 0.81 over 12,500 response pairs), and the Breast Cancer
Classification benchmark shows the calibration and threshold-analysis
discipline that turns raw model outputs into an actionable, reviewable
signal. Both are "make massive datasets legible to a human overseer"
projects — the team's stated mission.

## Cover letter (draft)

Dear AI Observability team,

I'm applying for the Research Engineer, AI Observability role because the
"use Claude to understand what's happening in the data" framing is the same
one I've been building on. My AI Safety Red-Team Evaluation project used an
LLM-judge ensemble to structure open-ended judgments about model responses
across 12,500 pairs and six harm categories, reported α = 0.81 and 96.8%
held-out accuracy, and — importantly for observability — kept uncertainty
as a first-class quantity that flows into the review workflow, not a
footnote at the end.

The Breast Cancer Classification benchmark and its calibration /
threshold / SHAP layer are how I think about the last mile: raw model
outputs are noise; the observability job is to shape them into a signal a
human oversight process can act on without over-trusting or dismissing it.

I'd like to build the pipeline side of that at Anthropic scale. My
statistics background (M.S., Applied Statistics, RIT) is why I default to
uncertainty-quantified reporting instead of dashboard maximalism.

I've also applied to the Model Evaluations role (#5198255008) — different
teams, complementary skill sets; happy to talk about which is the better
mutual fit.

— Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Tailored resume bullets

- Used an LLM-judge ensemble to structure open-ended labels over 12,500
  response pairs, reporting reliability (α = 0.81) and downstream classifier
  performance (96.8%) as separate quantities — an observability primitive,
  not just a benchmark score.
- Instrumented calibration curves, threshold sweeps, and SHAP attributions
  on a diagnostic benchmark — the shape of the signal a human oversight
  workflow needs to consume.
- Built productionizable eval and reporting artifacts with MLflow, PyMC,
  and standard HF tooling.

## Follow-ups

- If the current posting is SF-only, note openness to relocate or ask
  whether the NYC office is open for this req before submitting.
- Cross-reference with the Model Evaluations application (#01) — mention
  each to the recruiter to avoid duplicate routing.
