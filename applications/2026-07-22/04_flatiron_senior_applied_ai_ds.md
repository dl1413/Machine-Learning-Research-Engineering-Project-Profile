# Flatiron Health — Senior Applied AI Data Scientist

- **Location:** New York, NY (hybrid, 3 days on-site)
- **Lead project:** Breast Cancer Classification
- **Supporting projects:** AI Safety Red-Team, LLM Ensemble Bias Detection
- **JD:** https://www.builtinnyc.com/job/senior-applied-ai-data-scientist/8615163
- **Portfolio:** https://github.com/dl1413/machine-learning-research-engineering-project-profile

---

## Cover Letter

Dear Flatiron Health Applied AI team,

I'm applying for the Senior Applied AI Data Scientist role because
Flatiron's core problem — turning unstructured oncology notes into
structured, defensible research data via LLMs — sits exactly where my
three portfolio projects overlap: clinical-grade modeling, LLM
ensemble evaluation, and reproducible reporting for regulated
audiences.

The **Breast Cancer Classification** project is the direct oncology
signal. **99.12% accuracy, 100% precision (zero false positives),
98.59% recall, ROC-AUC 0.9987** on the Wisconsin dataset with an
8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking,
Voting). More useful than the headline number: **Platt-calibrated
probabilities (ECE 0.0089, a 71.5% reduction from raw)** and
**context-adaptive thresholds (100% sensitivity at 0.31 for mass
screening)** — the calibration and decision-policy layer that makes a
model actually deployable in a clinical setting. SHAP-based
explanations for clinician trust, IEEE 2830-2025 model card for
governance.

The Flatiron-specific twist — LLM extraction over oncology notes — is
where my other two projects come in:

- **AI Safety Red-Team Evaluation**: dual-stage LLM ensemble
  (GPT-4o, Claude-3.5, Llama-3.2) → stacking classifier, 12,500
  response pairs annotated at **α = 0.81 inter-rater reliability**
  and **$0.018/sample vs. $6.12 human (340× cost reduction)**. This
  is the pattern you'd use to scale note-abstraction quality checks
  without giving up audit reliability.
- **LLM Bias Detection**: 67,500 ratings, Bayesian hierarchical
  model with MCMC diagnostics (R-hat < 1.01). Publisher-level bias
  detection at 95% HDI. Same idea, applied to LLM outputs where
  the ground truth is contested — which is exactly what happens when
  three LLMs disagree on a molecular abstraction.

I've published 3 technical reports with reproducibility artifacts,
IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act alignment — the sort
of documentation a Roche-affiliated Applied AI team is likely to
value at pilot-to-production time.

Best,
Derek Lankeaux
MS Applied Statistics, Rochester Institute of Technology (2026)
LinkedIn: https://linkedin.com/in/derek-lankeaux

---

## One-Page Pitch

Flatiron's product is defensible clinical data from messy oncology
notes, at scale. The two hard parts of that are (a) LLM output
quality that survives clinician + regulator scrutiny, and (b)
extraction models that stay calibrated across sites, cohorts, and
years. My red-team project is a study in (a): three-LLM ensemble
grading at α = 0.81 at 850 samples/hour with a full audit trail. My
breast-cancer project is a study in (b): the calibration and
threshold-policy work that turns a 99% classifier into a screening
tool with 100% sensitivity at a defined decision threshold, plus
SHAP for the clinician conversation. Both are shipped with
publication-grade reports, MLflow tracking, and IEEE / ISO / EU AI
Act artifacts. The gap I'd need to close on day one is
oncology-specific abstraction (ICD, mCODE, LOINC, real-world data
provenance) — that's a ramp, not a foundation.

---

## Follow-up plan

- **T+0:** Submit via Flatiron careers portal (Built In NYC link
  redirects there).
- **T+2 days:** Look up the Applied AI team on Flatiron's engineering
  blog; note any authors working on note-abstraction LLM evals.
- **T+7 days:** LinkedIn note to a team member with a specific
  question about their multi-LLM disagreement handling. Reference
  the α = 0.81 result and how it might apply.
- Interview prep: real-world oncology data (mCODE), LLM evaluation
  patterns for clinical NLP, HIPAA and data-provenance basics.
