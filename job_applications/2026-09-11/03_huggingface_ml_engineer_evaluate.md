# 03 · Hugging Face — ML Engineer, Evaluate

- **Company:** Hugging Face
- **Role:** ML Engineer — Evaluate (evaluation library and harness)
- **Location:** Remote (US); NYC office available
- **Application URL (from search):** https://apply.workable.com/huggingface/j/66C7B15E3D
- **Priority:** ⭐⭐ (open-source eval library work — direct alignment)

## Why this fits Derek

The Evaluate team owns the shared evaluation surface a lot of the open-source
LLM ecosystem builds on top of. Two of Derek's three primary projects are
end-user evidence of what that surface needs: the AI Safety Red-Team
Evaluation shows why a harness has to keep annotation reliability separate
from downstream accuracy (α = 0.81 and 96.8% held-out are two different
numbers, not one), and the LLM Bias Detection project shows what falls out
when you scale rubric-based judging to 67,500 ratings and need proper
partial-pooling to interpret rater disagreement. Both projects use MLflow
for tracking, PyMC for uncertainty, and standard Hugging Face tooling in the
model layer.

## Cover letter (draft)

Dear Hugging Face Evaluate team,

I'm applying because most of the evaluation work I've published this year
depends on the primitives your team owns, and I'd rather help build them
than route around them. My AI Safety Red-Team Evaluation project ran a
two-stage LLM-ensemble + supervised harm classifier over 12,500 response
pairs across six harm categories and reported α = 0.81 alongside 96.8%
held-out accuracy — two numbers on purpose, because reliability and
performance are different measurements. My LLM Ensemble Textbook Bias
Detection project scaled a rubric-based judge to 67,500 ratings across
4,500 passages, hit α = 0.84, and used PyMC partial pooling with proper
MCMC diagnostics to model publisher-level effects.

The gap I keep hitting is on the harness side: sharing eval definitions
across studies without hand-rolling glue, tracking rater-level and
item-level variance in first-class metrics, and letting a run report both
a point estimate and its uncertainty by default. That's the work I want
to contribute to on Evaluate. I'm US-authorized and comfortable remote.

Full technical reports are linked below.

— Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Tailored resume bullets

- Consumed Hugging Face + MLflow + PyMC across two independent LLM-eval
  studies totaling ~80,000 rated items, with reliability and uncertainty
  reported separately from accuracy.
- Documented and shipped project reports in the same "problem framing → eval
  design → limitations" format Evaluate encourages contributors to adopt.
- Built rubric-based LLM-judge pipelines and inter-rater reliability
  reporting compatible with a standard harness surface.

## Follow-ups

- If Workable URL is stale, cross-check at huggingface.co/careers.
- Include GitHub links (this repo + LLM-Portfolio) — HF weighs public work.
