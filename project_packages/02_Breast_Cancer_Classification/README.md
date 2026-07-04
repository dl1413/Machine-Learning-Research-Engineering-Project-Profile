# Project 02 — Breast Cancer ML Classification

**Author:** Derek Lankeaux, MS Applied Statistics
**Date:** 2026
**Compliance:** IEEE 2830-2025 · ISO/IEC 23894:2025 · EU AI Act 2025

## Publication

| Document | File |
|----------|------|
| Technical Report (PDF) | [`Breast_Cancer_Classification_Publication.pdf`](./Breast_Cancer_Classification_Publication.pdf) |

## Summary

Clinical-grade ensemble ML system benchmarked on the Wisconsin Diagnostic Breast Cancer dataset, exceeding human expert performance on all primary metrics.

**Key Results:**
- 99.12% accuracy (AdaBoost, best-in-class)
- 100% precision — zero false positives
- 98.59% recall — minimal missed cases
- ROC-AUC: 0.9987 (near-perfect discrimination)
- Platt calibration reduces ECE by 71.5% (0.0312 → 0.0089)

**Tech Stack:** `scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `Optuna` `SMOTE` `SHAP` `MLflow` `FastAPI`
