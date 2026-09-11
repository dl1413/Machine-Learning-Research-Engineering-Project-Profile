# 04 · Arize AI — AI Application Engineer

- **Company:** Arize AI
- **Role:** AI Application Engineer (LLM observability platform)
- **Location:** Remote (US)
- **Application URL (from search):** https://himalayas.app/companies/arize-ai/jobs/ai-application-engineer-7915873090 (cross-check at arize.com/careers)
- **Priority:** ⭐⭐ (observability company — direct project alignment)

## Why this fits Derek

Arize is an LLM/ML observability platform. Two of Derek's three primary
projects operationalize what observability has to catch: the AI Safety
Red-Team Evaluation shows how to design a two-stage judge that flags harm
categories and reports uncertainty per prediction, and the Breast Cancer
Classification benchmark shows the calibration and threshold-analysis
discipline that turns model drift into an actionable alert instead of a
noisy line on a dashboard. The LLM Bias Detection work adds the
partial-pooling / hierarchical-modeling piece that makes drift comparisons
across content sources statistically honest.

## Cover letter (draft)

Dear Arize hiring team,

I'm applying for the AI Application Engineer role because the projects I've
built this year are exactly the kind of downstream evaluation your platform
is designed to make legible. My AI Safety Red-Team Evaluation combined an
LLM-judge ensemble with a supervised harm classifier over 12,500 response
pairs, reported α = 0.81 and 96.8% held-out accuracy, and — importantly for
observability — kept per-prediction uncertainty as a reportable quantity
alongside the point estimate. My Breast Cancer Classification benchmark on
WDBC (99.12% held-out, 0.9987 ROC-AUC) is a case study in what calibration,
threshold analysis, and SHAP-based attribution have to look like before a
model's outputs are safe to act on — the same discipline an observability
platform's users need instrumented for them.

My LLM Bias Detection project used PyMC partial pooling to model
publisher-level effects on 67,500 ratings across 4,500 passages; that
statistical machinery is what makes cross-cohort drift comparisons honest
rather than misleading.

I'd bring both the eval-design perspective and the willingness to write the
customer-facing integration code that makes it usable at customer time.
Remote works for me.

— Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Tailored resume bullets

- Built LLM-judge + supervised harm classification pipeline with
  per-prediction uncertainty (α = 0.81; 96.8% held-out over 12,500 pairs)
  — the exact signal an observability platform surfaces to customers.
- Instrumented calibration curves, threshold sweeps, and SHAP attributions
  on a diagnostic benchmark (99.12% / 0.9987 ROC-AUC) — the visualization
  surface Arize customers ask for.
- Modeled publisher-level effects on 67,500 LLM ratings using PyMC partial
  pooling, giving statistically honest cross-cohort drift comparisons.

## Follow-ups

- Verify posting on arize.com/careers — Himalayas mirrors can lag.
- Mention observability specifically in the free-text field; it's the axis
  Arize screens on.
