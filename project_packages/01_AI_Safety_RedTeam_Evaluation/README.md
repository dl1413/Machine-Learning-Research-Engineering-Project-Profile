# Project 01 — AI Safety Red-Team Evaluation

**Author:** Derek Lankeaux, MS Applied Statistics
**Date:** 2026
**Compliance:** IEEE 2830-2025 · ISO/IEC 23894:2025 · EU AI Act 2025

## Publication

| Document | File |
|----------|------|
| Technical Report (PDF) | [`AI_Safety_RedTeam_Evaluation_Publication.pdf`](./AI_Safety_RedTeam_Evaluation_Publication.pdf) |

## Summary

Automated harm-detection framework using a dual-stage LLM ensemble and ML classification pipeline evaluated on 12,500 AI response pairs across 6 harm categories.

**Key Results:**
- 96.8% accuracy (Stacking Classifier: 97.2% precision, 96.1% recall)
- 340× cost reduction: $0.018/sample vs. $6.12 human annotation
- Krippendorff's α = 0.81 (excellent ensemble reliability)
- Dual-filter reduces harm rate from 21.8% → 4.8% (78% reduction)

**Tech Stack:** `GPT-4o` `Claude-3.5` `Llama-3.2` `XGBoost` `Stacking` `PyMC` `SHAP` `MLflow`
