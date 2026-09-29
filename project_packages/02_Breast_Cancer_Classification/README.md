# Breast Cancer Classification Benchmark

**Type:** Independent technical case study
**Focus:** Ensemble-learning evaluation and error analysis on diagnostic benchmark data
**Origin:** Extends a two-semester graduate capstone at RIT on the same dataset
**Report version:** 4.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../Breast_Cancer_Classification_Report.md) |
| Publication-formatted report | [PDF](../../Breast_Cancer_Classification_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Classify breast masses as benign or malignant from 30 cell-nucleus measurements taken from digitized fine needle aspirate images (WDBC). | Preprocessing, feature selection, calibration, threshold analysis, SHAP explanations, and eight-model benchmarking. | 569 samples; 99.12% held-out accuracy (113/114); all 43 malignant test cases detected; 0.9987 ROC-AUC. |

## What this demonstrates

- Error analysis at the level of individual cases: the single test error is a
  benign tumor flagged as malignant, and no malignant case was missed.
- Diagnostic metrics reported with malignancy as the positive class, plus
  calibration and threshold analysis alongside headline accuracy.
- Per-prediction SHAP explanations tying each call to the nuclear features
  that drove it.
- A clear account of why a small, curated, single-center benchmark of
  pre-extracted features is not evidence of clinical readiness.

## Scope

This is an educational benchmark on the Wisconsin Diagnostic Breast Cancer
dataset. It is not a clinical device, a patient-validation study, medical
advice, or evidence for independent clinical use.

## Methods

`scikit-learn` · `XGBoost` · `LightGBM` · `AdaBoost` · `Optuna` · `SMOTE` · `SHAP`
