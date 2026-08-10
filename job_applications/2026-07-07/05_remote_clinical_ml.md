# 05 — Remote Clinical ML Data Scientist (Tempus AI / Flatiron Health / Komodo)

**Location:** Fully remote (US)
**Target requisitions (verify live before applying):**
- Tempus AI — Data Scientist, Oncology ML: https://www.tempus.com/careers/
- Flatiron Health — Quantitative Scientist, Oncology: https://flatiron.com/careers/
- Komodo Health — Data Scientist, Real-World Evidence: https://www.komodohealth.com/careers

Pick whichever requisition matches the day's search filter — the cover letter below is written generically enough to swap the company name.

## What these teams want

> Clinical-grade ML with calibrated, decision-ready outputs; real-world data comfort; ability to defend a model to a clinical stakeholder; ideally, some LLM / GenAI experience for the emerging "clinical documentation" and "structured-abstraction" work.

## Why this candidate fits

Derek's **Breast Cancer ML Classification** is the anchor — clinical benchmark, calibration story, threshold policy, SHAP explainability, IEEE 2830-2025 fairness auditing. The two LLM projects layer on the increasingly-expected GenAI capability that healthcare-DS teams are hiring for now (structured abstraction from clinical notes, LLM-assisted labeling).

## Project → Requirement mapping

| Clinical-ML requirement | Anchor project | Evidence |
|---|---|---|
| Clinical-grade model performance | Breast Cancer ML | 99.12% acc, 100% precision, ROC-AUC 0.9987 |
| Calibration for clinical decisions | Breast Cancer ML | Platt scaling ECE 0.0312 → 0.0089 (71.5% reduction) |
| Threshold policy for screening | Breast Cancer ML | Context-adaptive; 100% sensitivity at t=0.31 for screening |
| Fairness / responsible AI | Breast Cancer ML + Red-Team | SHAP fairness audit, IEEE 2830-2025 |
| LLM-assisted labeling of clinical text | AI Safety Red-Team | 340× cheaper than human, α = 0.81 pattern transfers directly |
| Uncertainty for cohort-level decisions | LLM Bias Detection | PyMC hierarchical, HDI, bootstrap CIs |

## Cover letter

> Dear [Company] team,
>
> I am applying for the remote Data Scientist role because I want to bring a statistician's clinical-decision framing to a healthcare-ML team that is scaling.
>
> My clinical-grade Breast Cancer ML Classification project benchmarked 8 algorithms and shipped an AdaBoost ensemble at 99.12% accuracy, 100% precision, and 98.59% recall on the diagnostic task (ROC-AUC 0.9987). More importantly for a clinical setting, I built the decision layer: Platt scaling reduced expected calibration error 71.5% (0.0312 → 0.0089), and context-adaptive thresholds preserve 100% sensitivity at t = 0.31 for a mass-screening policy while a different operating point serves confirmatory workflows. SHAP-based fairness auditing follows IEEE 2830-2025.
>
> Healthcare-ML teams are increasingly using LLMs for structured abstraction and labeling, and I have that capability too. My AI Safety Red-Team Evaluation project built a dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that annotates response pairs at Krippendorff's α = 0.81, then trains a Stacking Classifier on 47 engineered features — 96.8% accuracy, 850 samples/hour, 340× cheaper than human annotation. The same pattern transfers directly to abstracting oncology charts or pathology reports at scale.
>
> My LLM Ensemble Bias Detection work adds cohort-level uncertainty quantification: PyMC hierarchical modeling with R-hat < 1.01 and 95% HDI, plus Friedman χ² = 42.73 (p < 0.001) for significance testing across groups — the toolkit for real-world-evidence analyses.
>
> All 3 projects ship with MLflow, FastAPI (<100ms p95 for the clinical model), and IEEE 2830-2025 / ISO 23894 / EU AI Act alignment. MS in Applied Statistics (RIT, 2026), US work authorized, available for remote / hybrid.
>
> Thank you for reading.
>
> Derek Lankeaux · linkedin.com/in/derek-lankeaux · github.com/dl1413

## Screener prep

- Expect to defend the ECE metric choice and how you'd monitor drift on a clinical model.
- Expect: "if a clinician disagrees with the model, what do you do?" → threshold-policy story + SHAP walkthrough.
- Real-world data (RWD) fluency: mention OMOP CDM, ICD-10, LOINC if you have exposure; if not, be honest and pivot to methodology.

## Resume tweaks

- Move Breast Cancer ML to the top of the projects section for these applications.
- Add one line: "Deployment target: FastAPI microservice with <100ms p95 latency, MLflow model registry" — clinical-ML hiring reads deployment carefully.
