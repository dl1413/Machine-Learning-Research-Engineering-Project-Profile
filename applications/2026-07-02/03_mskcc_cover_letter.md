# Cover Letter - Memorial Sloan Kettering, Data Scientist, Radiology

**Location:** New York, NY
**Lead project:** Clinical-Grade Breast Cancer ML Classification
**JD link:** https://www.linkedin.com/jobs/view/data-scientist-radiology-at-memorial-sloan-kettering-cancer-center-4097062258

---

Dear MSK Radiology Data Science team,

The work I want to point to is a clinical-grade breast-cancer classifier
I shipped this year - directly relevant to the Radiology Data Scientist
role. I benchmarked 8 algorithms end-to-end (Random Forest, XGBoost,
LightGBM, AdaBoost, Stacking, Voting, plus two more) on the Wisconsin
Diagnostic Breast Cancer dataset and landed at **99.12% accuracy with
100% precision (zero false positives), 98.59% recall, and ROC-AUC
0.9987** - comfortably above the 90-95% range typically cited for
human expert reads.

Because it's a clinical decision-support setting, model performance was
only half the work. The pipeline also had to be **explainable and
auditable**:

- **SHAP explanations per prediction**, so a radiologist can see *why*
  the model flagged a case, not just *that* it did.
- **VIF-pruned features** and **Platt-calibrated probabilities**
  (ECE reduced 71.5%, from 0.0312 to 0.0089), so downstream thresholds
  correspond to real risk.
- **Context-adaptive thresholds** - e.g., a 100%-sensitivity operating
  point at 0.31 for mass screening versus a higher-precision point for
  confirmation.
- **IEEE 2830-2025 transparency alignment**, MLflow model registry,
  FastAPI service at <100ms p95 - the deployment pattern that survives
  clinical integration and IRB review.

I'd bring two other things MSK's mission cares about:

- **Statistical training.** I'm finishing an MS in Applied Statistics at
  RIT (2026), specializing in Bayesian methods, MCMC, and experimental
  design - useful when a radiology workflow needs prospective cohort
  analysis or calibration monitoring rather than a one-shot benchmark.
- **Complementary rigor from adjacent projects.** Two more published
  reports (multi-LLM bias detection with PyMC hierarchical modeling and
  R-hat < 1.01; LLM red-team evaluation at Krippendorff's alpha =
  0.81) show the same "measure carefully, ship reproducibly" habit
  applied to different domains.

I'm based in / available for New York City and targeting a 2026 start.
Portfolio, code, and three technical reports are on GitHub at dl1413.
Happy to walk through the WBCD project - the clinical-grade
classifier is the fastest way to see how I'd approach the Radiology
team's problems.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
