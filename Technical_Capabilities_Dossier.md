# Technical Capabilities Dossier: ML Research Engineering & Applied Statistics

**Author:** Derek Lankeaux, MS (Applied Statistics)
**Institution:** Rochester Institute of Technology — Independent Research Portfolio
**Date:** June 2026
**Project:** Synthesis of three published technical reports — AI Safety Red-Team Evaluation, LLM-Ensemble Bias Detection, and Clinical-Grade Predictive Modeling
**AI Standards Compliance:** IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act (high-risk system documentation)

---

## Abstract

This dossier consolidates the technical methodology behind three independently published research projects into a single advanced reference for technical reviewers and hiring panels. It documents (i) a dual-stage large-language-model (LLM) ensemble evaluation system that auto-grades 12,500 model-response pairs across six harm categories at 96.8% accuracy (ROC-AUC 0.9923) and a 340x reduction in per-sample annotation cost, (ii) a Bayesian hierarchical inference pipeline that processes 67,500 LLM bias ratings over 4,500 passages and localizes statistically significant publisher-level effects (Friedman chi-squared = 42.73, p < 0.001) while maintaining MCMC convergence (R-hat < 1.01, ESS > 1000), and (iii) a calibrated 8-algorithm predictive-modeling benchmark reaching 99.12% accuracy with 100% precision (ROC-AUC 0.9987, expected calibration error 0.0089) served behind a sub-100ms p95 inference API. Across all three systems the emphasis is on defensible measurement: explicit uncertainty quantification (95% highest-density intervals, bootstrap confidence intervals), inter-rater reliability controls (Krippendorff's alpha = 0.81-0.84), multiple-comparison correction, calibration, and audit-grade provenance. The document is intended to be read alongside the three full technical reports and accompanying source code.

**Keywords:** LLM evaluation, red-teaming, Bayesian hierarchical modeling, MCMC diagnostics, inter-rater reliability, Krippendorff's alpha, stacking ensemble, model calibration, SHAP explainability, MLOps, FastAPI inference, IEEE 2830-2025, ISO/IEC 23894:2025

---

## 1. Scope and Design Philosophy

Three projects, one operating principle: **a result is only as strong as the evidence that defends it.** Each system below was built so that its headline number survives an adversarial technical review — point estimates are paired with uncertainty intervals, agreement metrics accompany every multi-rater claim, and every reported figure is traceable to a committed, reproducible pipeline.

| Project | Problem class | Primary technical contribution | Headline result |
|---|---|---|---|
| AI Safety Red-Team Evaluation | LLM-as-judge harm classification | Dual-stage ensemble + stacking meta-classifier | 96.8% acc, ROC-AUC 0.9923, 340x cost reduction |
| LLM-Ensemble Bias Detection | Hierarchical statistical inference | Partial-pooling Bayesian model over publishers | Friedman chi-squared = 42.73, p < 0.001; alpha = 0.84 |
| Clinical-Grade Classification | Calibrated predictive modeling | 8-algorithm benchmark + probability calibration | 99.12% acc, 100% precision, ECE = 0.0089 |

The remainder of this dossier describes the architecture (Section 2), the statistical methodology (Section 3), the predictive-modeling and calibration stack (Section 4), the production/MLOps engineering (Section 5), the responsible-AI controls (Section 6), and reproducibility (Section 7), and closes with a competency matrix (Section 8).

---

## 2. System Architecture — Dual-Stage LLM Evaluation

### 2.1 Two-stage decomposition

The red-team evaluator separates *judgment* from *classification*. Stage 1 runs an ensemble of three frontier LLMs (GPT-4o, Claude-3.5-Sonnet, Llama-3.2) as independent red-team graders; Stage 2 reconciles their (often disagreeing) labels with a supervised meta-classifier trained on engineered harm-signal features. This decomposition is what allows the system to be both *cheap* (LLM passes are batched and cached) and *defensible* (a calibrated classifier, not a single model's opinion, produces the final label).

```
                          12,500 response pairs
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         GPT-4o judge      Claude-3.5 judge     Llama-3.2 judge      ← Stage 1: ensemble grading
              │                   │                   │                (async, circuit-breakered,
              └─────────┬─────────┴─────────┬─────────┘                 850 samples/hr)
                        ▼                   ▼
              47 engineered features   inter-rater reliability
              (linguistic/semantic/    (Krippendorff's alpha = 0.81)
               structural)
                        │
                        ▼
            Stacking meta-classifier (XGBoost base + logistic meta) ← Stage 2: reconciliation
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       harm label + p(harm)   SHAP attribution + audit record
       (97.2% precision)      (IEEE 2830-2025 provenance)
```

### 2.2 Feature engineering

The meta-classifier consumes **47 engineered features** spanning three families:

- **Linguistic** — lexical risk markers, imperative/instructional mood, refusal-evasion phrasing, obfuscation/encoding signals.
- **Semantic** — topical proximity to the six harm categories (dangerous information, hate, deception, privacy violation, illegal activity, self-harm), intent vs. content separation.
- **Structural** — multi-turn escalation patterns, response length/format anomalies, jailbreak-template fingerprints.

An adversarial-attack taxonomy aligned to MITRE ATLAS organizes the inputs; multi-turn escalation was identified as the highest-risk vector (31.8% of successful attacks), and a dual-filter defense reduced the observed harm rate from 21.8% to 4.8% (a 78% relative reduction).

### 2.3 Throughput and cost engineering

The pipeline sustains **850 samples/hour** at **$0.018/sample** — a **340x** reduction against a $6.12 human-annotation baseline — without sacrificing reliability (Krippendorff's alpha = 0.81 across the three judges). The cost/throughput envelope comes from async batched API calls, response caching, circuit breakers, and exponential backoff (Section 5), not from cutting evaluation rigor.

### 2.4 Classification performance

On the 12,500-pair benchmark the Stage-2 stacking classifier reaches **96.8% accuracy, 97.2% precision, 96.1% recall, and ROC-AUC 0.9923**. Precision is deliberately prioritized: in a red-team setting a false "safe" label is costlier than a false "harmful" flag routed to human review.

---

## 3. Statistical Methodology — Bayesian Hierarchical Inference

The bias-detection project is the statistical centerpiece: 67,500 LLM bias ratings over 4,500 textbook passages (2.5M tokens) across five publishers, analyzed to answer a causal-flavored question — *does measured bias differ by publisher beyond what passage-level noise explains?*

### 3.1 Why hierarchical, why Bayesian

Passages are nested within publishers and topics. A naive per-publisher mean ignores both the unequal evidence per publisher and the shared structure across them. A **partial-pooling hierarchical model** (PyMC) shrinks noisy publisher estimates toward the grand mean in proportion to their uncertainty, and a fully Bayesian treatment yields **95% highest-density intervals (HDI)** per publisher rather than fragile point estimates — the difference between "publisher X scored higher" and "publisher X is credibly more biased."

```
  μ_global  ~  Normal(0, σ_global)                       (grand mean)
  σ_pub     ~  HalfNormal(·)                             (between-publisher scale)
  θ_pub[j]  ~  Normal(μ_global, σ_pub)        j = 1..5   (publisher effects, partially pooled)
  y[i]      ~  Normal(θ_pub[ pub(i) ], σ_obs)            (passage-level ratings)
```

### 3.2 Convergence and reliability diagnostics

Inference is only trustworthy if the sampler converged and the raters agreed:

- **MCMC convergence:** R-hat < 1.01 on all parameters; effective sample size (ESS) > 1000 — no divergent-transition pathologies.
- **Inter-rater reliability:** Krippendorff's alpha = 0.84 with 92% pairwise correlation across GPT-4o / Claude-3.5 / Llama-3.2, establishing that the rating instrument itself is stable before any downstream claim is made.

### 3.3 Frequentist corroboration and multiple comparisons

The Bayesian result is cross-checked with a distribution-free omnibus test: a **Friedman test (chi-squared = 42.73, p < 0.001)** rejects the null of equal publisher distributions, followed by **post-hoc Nemenyi pairwise comparisons** to localize *which* publishers differ — controlling family-wise error across the comparison set rather than reporting uncorrected pairwise p-values. Three of five publishers show credible directional bias. A cross-topic analysis flags Social Issues as the most polarized dimension (Δ = 1.36 points across publishers), and a Spearman correlation matrix surfaces structural editorial relationships (ρ up to 0.74).

### 3.4 Uncertainty routing

Bootstrap confidence intervals at the passage level identify the **12.3% highest-uncertainty passages**, which are routed to human expert review. This converts uncertainty from a footnote into an operational signal — the model decides *what it is unsure about* and escalates accordingly.

---

## 4. Predictive Modeling, Calibration, and Serving

The clinical-grade classifier demonstrates disciplined supervised modeling on tabular data under a high-stakes decision policy.

### 4.1 Preprocessing and selection

- **VIF-based multicollinearity pruning** removes redundant predictors that destabilize coefficient estimates and feature attributions.
- **SMOTE** addresses class imbalance on the training folds only (never the held-out test set, to avoid optimistic leakage).
- **Recursive feature elimination (RFE)** with stratified cross-validation selects a compact, stable feature subset.

### 4.2 Benchmark and winner selection

Eight algorithms — Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting, a Voting ensemble, a Stacking ensemble, and a regularized linear baseline — are benchmarked under cross-validation with identical preprocessing. Model choice is by evidence, not fashion: the winning ensemble reaches **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987**, above the 90-95% range typically cited for human expert reads.

### 4.3 Probability calibration

Raw classifier scores are not probabilities. The pipeline applies **Platt / isotonic calibration** and reports an **expected calibration error (ECE) of 0.0089**, so that a predicted 0.9 means roughly 90% empirical positive rate — a prerequisite for threshold policies. Decision thresholds are tuned to the operating context (e.g., a sensitivity-maximizing threshold for screening), decoupling the calibrated probability from the deployment decision rule.

### 4.4 Serving

The winning model is productionized behind a **FastAPI** service with **sub-100ms p95 latency**, an MLflow model registry for versioned promotion, and per-prediction SHAP explanations attached to each response.

---

## 5. Production & MLOps Engineering

All three systems share a resilient data/serving substrate built to run unattended at scale (80K+ API calls, 2.5M tokens):

- **Resilience:** circuit breakers and exponential backoff isolate provider-side failures; async batching maximizes throughput without tripping rate limits.
- **Experiment tracking & lineage:** MLflow captures parameters, metrics, and artifacts for every run, giving each reported number a reproducible provenance trail.
- **Serving:** containerized FastAPI endpoints, MLflow model registry for staged promotion, sub-100ms p95 inference.
- **Monitoring:** drift detection and reliability monitoring on inputs and outputs, with high-uncertainty cases routed to human review.

```
 ingest ─▶ async batch + cache ─▶ circuit breaker / backoff ─▶ model(s) ─▶ MLflow tracking
                                                                   │
                                          registry ◀── promotion ──┤
                                                                   ▼
                                              FastAPI (<100ms p95) ─▶ SHAP + audit record ─▶ drift monitor
```

---

## 6. Responsible AI and Compliance

The portfolio treats governance as an engineering requirement, not paperwork appended afterward:

- **Explainability:** SHAP attributions per prediction (and per harm-signal feature in the eval system), enabling case-level review.
- **Documentation standards:** artifacts aligned to **IEEE 2830-2025** (transparent ML), **ISO/IEC 23894:2025** (AI risk management), and **EU AI Act** high-risk documentation expectations — model cards, calibration plots, audit trails, and provenance.
- **Fairness & auditing:** reliability and bias diagnostics surfaced as first-class metrics; uncertainty routed to human oversight rather than silently absorbed.

These controls are increasingly prerequisites — not differentiators — for ML touching regulated domains (healthcare, finance, education, safety).

---

## 7. Reproducibility and Results Summary

Every figure in this dossier is traceable to one of three committed, reproducible pipelines with full technical reports and source code. Consolidated results:

| Metric | Red-Team Eval | Bias Detection | Clinical Classifier |
|---|---|---|---|
| Scale | 12,500 pairs / 6 categories | 67,500 ratings / 4,500 passages / 2.5M tokens | 8-algorithm benchmark |
| Primary metric | 96.8% accuracy | Friedman chi-sq = 42.73 (p < 0.001) | 99.12% accuracy |
| Discrimination | ROC-AUC 0.9923 | — | ROC-AUC 0.9987 |
| Precision / Recall | 97.2% / 96.1% | — | 100% / 98.59% |
| Reliability | alpha = 0.81 | alpha = 0.84, R-hat < 1.01 | ECE = 0.0089 |
| Uncertainty | 95% HDI (Bayesian risk) | 95% HDI + bootstrap CIs | calibrated p, threshold policy |
| Efficiency | $0.018/sample (340x) | 80K+ calls, MLflow lineage | <100ms p95 serving |

---

## 8. Technical Competency Matrix

| Area | Tools & methods |
|---|---|
| **Languages** | Python 3.12+, R, SQL, Bash |
| **Bayesian / inference** | PyMC, ArviZ, NumPyro, Stan; hierarchical models, MCMC diagnostics (R-hat, ESS), 95% HDI, partial pooling |
| **Experimentation / stats** | A/B testing, power analysis, Friedman / Nemenyi, multiple-testing correction (Bonferroni, FDR), bootstrap CIs, effect sizes, inter-rater reliability (Krippendorff's alpha, Cohen's kappa) |
| **ML / modeling** | scikit-learn, XGBoost, LightGBM, CatBoost, AdaBoost; stacking/voting ensembles, RFE, SMOTE, VIF, Platt/isotonic calibration, threshold tuning |
| **GenAI / LLM eval** | GPT-4o, Claude-3.5, Llama-3.2; LLM-as-judge, multi-model ensembles, prompt iteration, offline benchmarking, cost/latency analysis |
| **MLOps / serving** | MLflow, Weights & Biases, DVC, FastAPI, Docker, Kubernetes; circuit breakers, backoff, drift monitoring |
| **Responsible AI** | SHAP, LIME, Captum; model cards, IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act |

---

## References (selected)

1. Lankeaux, D. *AI Safety Red-Team Evaluation: Technical Analysis Report.* Independent Research, 2026.
2. Lankeaux, D. *LLM-Ensemble Textbook Bias Detection System: Technical Report.* Independent Research, 2026.
3. Lankeaux, D. *Clinical-Grade Breast Cancer ML Classification System: Technical Report.* Independent Research, 2026.
4. Krippendorff, K. *Content Analysis: An Introduction to Its Methodology.* SAGE.
5. Gelman, A., et al. *Bayesian Data Analysis,* 3rd ed. CRC Press.
6. Vehtari, A., et al. "Rank-normalization, folding, and localization: An improved R-hat." *Bayesian Analysis,* 2021.
7. IEEE 2830-2025 — Standard for Technical Framework and Requirements of Trusted ML.
8. ISO/IEC 23894:2025 — Artificial Intelligence — Guidance on Risk Management.
</content>
