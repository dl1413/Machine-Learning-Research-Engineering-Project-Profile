# 02 · EvolutionIQ — Senior Data Scientist, LLM Evaluation (Medhub)

- **Company:** EvolutionIQ
- **Role:** Senior Data Scientist — LLM Evaluation (Medhub product)
- **Location:** New York, NY or Remote
- **Application URL:** https://job-boards.greenhouse.io/evolutioniq/jobs/5748219004
- **Priority:** ⭐⭐⭐ (NYC/remote + LLM eval in a regulated domain)

## Why this fits Derek

Medhub is a healthcare-facing product. The AI Safety Red-Team Evaluation and
LLM Bias Detection projects both operationalize the pattern this role needs:
LLM judges producing structured ratings, annotation reliability measured
formally (Krippendorff's α, not accuracy-only), and Bayesian hierarchical
modeling used to separate signal from noise across raters and content
sources. The Breast Cancer Classification benchmark adds the calibration
and threshold-analysis discipline that healthcare-adjacent evaluation needs
before any recommendation reaches a clinician.

## Cover letter (draft)

Dear EvolutionIQ hiring team,

The Senior Data Scientist, LLM Evaluation role on Medhub is a natural fit
for the evaluation work I've been building. My AI Safety Red-Team Evaluation
project designed a two-stage LLM-ensemble + supervised classifier over
12,500 response pairs across six harm categories, hitting α = 0.81 and
96.8% held-out accuracy while keeping annotation reliability, classifier
performance, and uncertainty as three separate reported quantities. The LLM
Ensemble Textbook Bias Detection project scaled that pattern to 67,500
ratings across 4,500 passages, reached α = 0.84, and used PyMC partial
pooling to isolate publisher-level effects — the same statistical machinery
that generalizes to modeling rater and site effects in a healthcare-content
evaluation.

Because Medhub sits next to clinical decisions, I'd bring the same
calibration and threshold-analysis discipline I used in my Breast Cancer
Classification benchmark (99.12% held-out accuracy, 0.9987 ROC-AUC on WDBC,
with calibration curves and threshold sweeps reported alongside the
headline number). That work is explicitly framed as decision support rather
than a clinical device, and I'd carry the same framing into any LLM
evaluation whose output influences a clinical workflow.

I'm authorized to work in the US and open to NYC or remote. Portfolio and
full technical reports are linked below.

— Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

## Tailored resume bullets

- Built LLM-evaluation workflows in two independent studies (12,500 response
  pairs; 67,500 rubric ratings across 4,500 passages) with α = 0.81 and
  α = 0.84 respectively, keeping reliability separate from accuracy.
- Modeled publisher-level effects with PyMC partial pooling and MCMC
  diagnostics — same statistical pattern applies to rater / site effects in
  clinical-content evaluation.
- Applied calibration and threshold-sweep analysis on a diagnostic
  benchmark (WDBC, 99.12% / 0.9987 ROC-AUC) framed as decision support,
  not a clinical device.

## Follow-ups

- Confirm posting is still open (posted to Greenhouse; healthcare LLM eval
  is a fast-moving hire).
- If they ask for a work sample, send the LLM Bias Detection PDF plus the
  AI Safety PDF from `project_packages/`.
