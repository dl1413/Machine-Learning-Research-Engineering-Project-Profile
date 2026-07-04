# Project 03 — LLM Ensemble Textbook Bias Detection

**Author:** Derek Lankeaux, MS Applied Statistics
**Date:** 2026
**Compliance:** IEEE 2830-2025 · ISO/IEC 23894:2025 · EU AI Act 2025

## Publication

| Document | File |
|----------|------|
| Technical Report (PDF) | [`LLM_Bias_Detection_Publication.pdf`](./LLM_Bias_Detection_Publication.pdf) |

## Summary

Multi-LLM framework for detecting and quantifying political bias in educational textbooks, using a three-model ensemble and Bayesian hierarchical modeling for robust inference.

**Key Results:**
- 67,500 bias ratings across 4,500 textbook passages from 150 textbooks
- Krippendorff's α = 0.84 (excellent inter-rater reliability)
- Statistically significant publisher differences: Friedman χ² = 42.73, p < 0.001
- 3 of 5 publishers exhibit credible bias (95% HDI excluding zero)
- MCMC convergence: R-hat < 1.01, ESS > 3,000

**Tech Stack:** `GPT-4o` `Claude-3.5` `Llama-3.2` `PyMC` `ArviZ` `MLflow` `FastAPI` `LangChain`
