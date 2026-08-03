# Breast Cancer Classification Benchmark

**Type:** Independent technical case study
**Focus:** Ensemble-learning evaluation on diagnostic benchmark data
**Report version:** 4.0.0 · April 2026

| Read | Link |
|---|---|
| Full technical report | [Markdown](../../Breast_Cancer_Classification_Report.md) |
| Publication-formatted report | [PDF](./Breast_Cancer_Classification_Publication.pdf) |
| Portfolio overview | [README](../../README.md) |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| Compare ensemble methods for binary classification using the WDBC benchmark. | Preprocessing, feature selection, calibration, threshold analysis, and eight-model benchmarking. | 569 samples; 99.12% held-out accuracy; 0.9987 ROC-AUC; 10-fold cross-validation. |

## What this demonstrates

- A disciplined supervised-learning workflow from dataset inspection through
  cross-validation, calibration, and explainability.
- Why diagnostic metrics, thresholds, and uncertainty need to be considered
  alongside headline accuracy.
- A transparent basis for discussing model limitations and decision support.

## Scope

This is an educational benchmark on the Wisconsin Diagnostic Breast Cancer
dataset. It is not a clinical device, a patient-validation study, medical
advice, or evidence for independent clinical use.

## Methods

`scikit-learn` · `XGBoost` · `LightGBM` · `AdaBoost` · `Optuna` · `SMOTE` · `SHAP`
