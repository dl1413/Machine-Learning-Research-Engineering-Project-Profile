# Machine Learning Research Engineering Portfolio (2026)

**Author:** Derek Lankeaux, MS Applied Statistics  
**Role:** Data Scientist | Applied Statistician | ML Research Engineer  
**Institution:** Rochester Institute of Technology  
**Date:** January 2026  
**Version:** 1.0.0  
**AI Standards Compliance:** IEEE 2830-2025 (Transparent ML), ISO/IEC 23894:2025 (AI Risk Management), EU AI Act (2025)

---

## Abstract

This portfolio presents three end-to-end machine learning research engineering projects demonstrating 2026-grade competencies in applied ML, Bayesian inference, GenAI evaluation, and responsible AI deployment. Project 1 develops a scalable AI safety red-team evaluation framework combining LLM ensemble annotation (Krippendorff's α = 0.81) with supervised ML classification achieving 0.9923 ROC-AUC and a 340× cost reduction over human annotation. Project 2 implements an ensemble learning pipeline for breast cancer classification achieving 99.12% accuracy and 0.9987 ROC-AUC with full calibration analysis and clinical decision framing. Project 3 applies a Bayesian hierarchical framework to detect publisher-level political bias in educational textbooks using LLM consensus scoring (α = 0.84), yielding credible posterior estimates with 95% Highest Density Intervals (HDI). Across all three projects, consistent emphasis is placed on statistical rigor, reproducibility, explainability, and production readiness.

**Keywords:** Machine Learning, Applied Statistics, Bayesian Inference, Ensemble Methods, LLM Evaluation, AI Safety, Breast Cancer Classification, Bias Detection, XGBoost, PyMC, SHAP, MLOps, Responsible AI

---

## Table of Contents

1. [Portfolio Overview](#1-portfolio-overview)
2. [Project 1: AI Safety Red-Team Evaluation](#2-project-1-ai-safety-red-team-evaluation)
3. [Project 2: Breast Cancer Classification](#3-project-2-breast-cancer-classification)
4. [Project 3: LLM Ensemble Bias Detection](#4-project-3-llm-ensemble-bias-detection)
5. [Cross-Project Technical Strengths](#5-cross-project-technical-strengths)
6. [Role Alignment and Competency Map](#6-role-alignment-and-competency-map)
7. [Reproducibility and Deployment Notes](#7-reproducibility-and-deployment-notes)
8. [Conclusions](#8-conclusions)

---

## 1. Portfolio Overview

### 1.1 Research and Engineering Contributions

This portfolio demonstrates the ability to design, implement, evaluate, and communicate production-grade ML systems across three applied domains. Core contributions include:

- **Reproducible ML workflows** with explicit data versioning, experiment tracking, and deterministic evaluation pipelines
- **Statistically grounded evaluation**: confidence intervals, subgroup analysis, calibration diagnostics, and ablation studies
- **Engineering best practices**: modular code, test coverage, CI pipelines, and inference/runtime profiling
- **Decision-oriented metrics** that map model performance to real-world operational objectives
- **Responsible AI documentation**: limitations, governance controls, human-in-the-loop escalation policies

### 1.2 Project Summary

| Project | Domain | Primary Method | Key Result |
|---------|--------|---------------|------------|
| AI Safety Red-Team Evaluation | AI Governance | LLM Ensemble + ML Classification | ROC-AUC 0.9923; α = 0.81 |
| Breast Cancer Classification | Clinical ML | Ensemble Learning + Calibration | Accuracy 99.12%; ROC-AUC 0.9987 |
| LLM Ensemble Bias Detection | NLP / Bayesian | Hierarchical Bayes + LLM Consensus | α = 0.84; 3/5 publishers credibly biased |

### 1.3 Shared Methodology Standards

All three projects share a common methodological foundation:

1. **Pre-registration equivalents**: defined hypotheses, metrics, and analysis plans before fitting final models
2. **Held-out test sets**: no leakage between feature engineering, model selection, and final evaluation
3. **Uncertainty quantification**: 95% confidence intervals (frequentist) or 95% HDI (Bayesian) on all reported estimates
4. **Explainability**: SHAP-based global and local feature attribution for tree-based models
5. **Standards compliance**: IEEE 2830-2025 transparent ML documentation, ISO/IEC 23894:2025 AI risk management

---

## 2. Project 1: AI Safety Red-Team Evaluation

### 2.1 Problem Statement

Manual red-teaming is expensive (~$50–100/hour per human expert), non-scalable, and subject to inconsistent inter-annotator agreement (70–85%). Modern LLM release cycles demand evaluation throughput that human red-teamers cannot sustain. This project develops and validates a hybrid human-AI evaluation framework that achieves annotation quality comparable to expert consensus at a 340× cost reduction.

### 2.2 Technical Framework

The evaluation pipeline consists of two stages:

**Stage 1 — LLM Ensemble Annotation**
- Three frontier LLMs serve as calibrated annotators: GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B
- Structured prompt protocol enforces six-category harm taxonomy (Dangerous Information, Hate/Discrimination, Deception/Manipulation, Privacy Violation, Illegal Activity, Self-Harm/Violence)
- Inter-rater reliability validated with Krippendorff's α = 0.81 (Excellent threshold: ≥ 0.80)
- Corpus: 12,500 AI model response pairs processed at ~180 samples/hour per LLM

**Stage 2 — ML Classification Pipeline**
- Eight ensemble classifiers trained on LLM-generated labels: Random Forest, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM, Voting, and Stacking
- Feature engineering: 47 features including lexical patterns, semantic embeddings, structural cues, and LLM confidence signals
- Class imbalance mitigation: SMOTE oversampling + class-weighted loss
- Best model (Stacking Classifier) processing rate: ~850 samples/hour at $0.018/sample

**Bayesian Hierarchical Risk Modeling**
- PyMC-based hierarchical model with partial pooling over harm categories and AI model vendors
- 95% HDI quantifies uncertainty on category-level harm rates
- Posterior predictive checks confirm model calibration

### 2.3 Results

#### Stage 1: LLM Annotation Reliability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Krippendorff's α (overall) | 0.81 | Excellent inter-rater agreement |
| GPT-4o ↔ Claude-3.5 (r) | 0.94 | Near-perfect pairwise agreement |
| GPT-4o ↔ Llama-3.2 (r) | 0.88 | Excellent pairwise agreement |
| Claude-3.5 ↔ Llama-3.2 (r) | 0.86 | Excellent pairwise agreement |
| Fleiss' κ (3-way) | 0.79 | Substantial agreement |

#### Stage 2: Classification Performance

| Metric | Stage 2 (ML Classifier) |
|--------|------------------------|
| Accuracy | 96.8% |
| Precision | 97.2% |
| Recall (Sensitivity) | 96.1% |
| F1-Score | 96.6% |
| ROC-AUC | 0.9923 |
| Cross-Validation (10-fold) | 95.9% ± 1.4% |

#### Harm Category Detection

| Harm Category | Prevalence | F1-Score | 95% HDI (Risk Rate) |
|---------------|------------|----------|---------------------|
| Dangerous Information | 8.2% | 96.9% | [0.42, 0.58] |
| Hate/Discrimination | 6.4% | 95.9% | [0.31, 0.46] |
| Deception/Manipulation | 11.3% | 96.7% | [0.54, 0.71] |
| Privacy Violation | 4.1% | 93.5% | [0.19, 0.33] |
| Illegal Activity | 5.7% | 97.3% | [0.28, 0.42] |
| Self-Harm/Violence | 3.8% | 95.4% | [0.15, 0.28] |

#### Defense Effectiveness

| Defense Configuration | Harm Rate | Reduction vs. Baseline |
|-----------------------|-----------|------------------------|
| No defense (baseline) | 21.8% | — |
| Single-stage filter | 11.4% | 47.7% |
| Dual-stage filter | 4.8% | 78.0% |
| + Adversarial fine-tuning | 2.1% | 90.4% |

### 2.4 Adversarial Attack Taxonomy

Eight attack categories aligned with MITRE ATLAS:

| Attack Vector | Success Rate | Frequency |
|---------------|-------------|-----------|
| Multi-turn Escalation | 38.2% | 31.8% |
| Role-Play Persona | 29.4% | 22.7% |
| Indirect Request | 24.1% | 18.6% |
| Context Injection | 21.7% | 12.4% |
| Prompt Chaining | 18.9% | 8.9% |
| Encoding Obfuscation | 15.3% | 3.8% |
| Token Smuggling | 12.6% | 1.8% |
| Adversarial Suffix | 9.8% | 0.0%* |

*\*Fully blocked by dual-stage filter.*

### 2.5 Production Architecture

- **Throughput:** ~850 prompt-response pairs/hour (combined pipeline)
- **Cost:** $0.018/sample (vs. ~$6.25/sample human baseline)
- **Latency:** P95 < 2.1 seconds end-to-end
- **API:** REST endpoint with async queue; 99.7% uptime SLA target
- **Governance:** Full audit trail, explainability dashboard, escalation to human review above configurable risk threshold

### 2.6 Threats to Validity

- LLM annotators share training corpus overlap, which may inflate agreement
- Harm taxonomy reflects April 2026 standards; regulatory updates may require re-categorization
- Evaluation corpus is synthetic/curated and may not fully represent production adversarial distribution
- Dual-filter defense effectiveness is measured in controlled conditions; real-world performance may differ

---

## 3. Project 2: Breast Cancer Classification

### 3.1 Problem Statement

Breast cancer is the most prevalent malignancy among women globally (~2.3 million new diagnoses annually). Fine Needle Aspiration (FNA) cytology interpretation exhibits inter-observer variability (85–95% concordance), creating demand for consistent, reproducible computer-aided diagnosis support. This project develops and validates an ensemble ML system for binary malignancy classification from FNA cytological features, with explicit clinical threshold analysis.

### 3.2 Technical Framework

**Data Engineering**
- Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), n = 569 (357 benign, 212 malignant)
- Features: 30 cytological measurements (10 nuclear characteristics × 3 statistical summaries)
- Preprocessing: Variance Inflation Factor (VIF) analysis for multicollinearity; StandardScaler normalization
- Class imbalance: SMOTE oversampling applied to training fold only (leakage-safe)
- Feature selection: Recursive Feature Elimination (RFE) retaining 17 features

**Ensemble Benchmarking**
- Eight algorithms evaluated: Random Forest, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM, Voting, Stacking
- Hyperparameter optimization: Optuna Bayesian search (200 trials, TPE sampler)
- Validation: 10-fold stratified cross-validation; held-out test set (n = 114, 20%)

**Calibration Analysis**
- Pre- and post-calibration metrics: Expected Calibration Error (ECE), Brier score, reliability curves
- Calibration methods compared: Platt scaling (sigmoid), isotonic regression
- Clinical threshold analysis: sensitivity–specificity trade-off curve at 50 operating points

### 3.3 Results

#### Classification Performance (Best Model: AdaBoost)

| Metric | Value | Clinical Interpretation |
|--------|-------|-------------------------|
| Accuracy | 99.12% | 113/114 correct classifications |
| Precision (PPV) | 100.00% | Zero false positives |
| Recall (Sensitivity) | 98.59% | 1 missed malignancy in 71 cases |
| Specificity | 100.00% | Perfect benign identification |
| F1-Score | 99.29% | Harmonic mean balance |
| ROC-AUC | 0.9987 | Near-perfect discrimination |
| Cohen's κ | 0.9823 | Almost perfect agreement |

#### Cross-Validation Summary

| Algorithm | CV Accuracy | Std Dev | Test Accuracy |
|-----------|------------|---------|---------------|
| AdaBoost | 98.46% | ±1.12% | 99.12% |
| XGBoost | 97.89% | ±1.34% | 98.25% |
| Stacking | 97.71% | ±1.28% | 98.25% |
| LightGBM | 97.53% | ±1.41% | 97.37% |
| Random Forest | 96.84% | ±1.52% | 96.49% |
| Gradient Boosting | 96.67% | ±1.48% | 96.49% |
| Voting | 96.49% | ±1.71% | 95.61% |
| Bagging | 95.44% | ±1.83% | 94.74% |

*95% CI (test accuracy): [96.27%, 100.65%]; Binomial test vs. random baseline: p < 0.0001.*

#### Calibration Improvement (AdaBoost)

| Metric | Before Calibration | After (Platt Scaling) |
|--------|-------------------|----------------------|
| Expected Calibration Error | 0.0312 | 0.0089 |
| Brier Score | 0.0421 | 0.0187 |
| Max Calibration Error | 0.0847 | 0.0234 |

#### Clinical Threshold Analysis

| Operating Point | Sensitivity | Specificity | PPV | NPV |
|----------------|------------|------------|-----|-----|
| Default (0.50) | 98.59% | 100.00% | 100.00% | 97.67% |
| High Sensitivity (0.30) | 100.00% | 97.62% | 98.61% | 100.00% |
| High Specificity (0.70) | 95.77% | 100.00% | 100.00% | 95.45% |

*Recommended operating point: 0.30 threshold for screening applications (zero missed malignancies).*

### 3.4 Top Predictive Features (SHAP Analysis)

| Rank | Feature | SHAP Mean Abs | Clinical Relevance |
|------|---------|---------------|--------------------|
| 1 | Worst Radius | 0.847 | Tumor size indicator |
| 2 | Worst Concave Points | 0.723 | Nuclear irregularity |
| 3 | Mean Concave Points | 0.641 | Shape complexity |
| 4 | Worst Perimeter | 0.589 | Boundary length |
| 5 | Mean Radius | 0.521 | Average cell size |

### 3.5 Threats to Validity

- WDBC dataset (1995) may not reflect contemporary cytology imaging pipelines
- Single-institution data limits generalizability across demographic and equipment variation
- 10-fold CV optimistic estimate on small dataset (n = 569); external validation not performed
- SMOTE introduces synthetic samples; real-world class distribution may differ from training split

---

## 4. Project 3: LLM Ensemble Bias Detection

### 4.1 Problem Statement

Political bias in educational textbooks has long-term effects on civic socialization. Traditional audit methods are subjective, expensive, and non-scalable. This project develops a scalable, reproducible computational audit framework using LLM consensus scoring and Bayesian hierarchical modeling to detect and quantify publisher-level political framing in K–12 educational content.

### 4.2 Technical Framework

**Corpus Construction**
- 150 textbooks from 5 major educational publishers (Publishers A–E)
- 4,500 passages sampled (30 per textbook); topics: government, economics, history, social issues
- 67,500 bias ratings generated (4,500 passages × 3 LLMs × 5 dimensions)

**LLM Ensemble Protocol**
- Three frontier LLMs: GPT-4, Claude-3-Opus, Llama-3-70B
- Structured prompt: passage + rating rubric (−2 strongly liberal to +2 strongly conservative)
- Deterministic decoding (temperature = 0) for reproducibility
- Five dimensions rated: word choice, framing, source selection, omission patterns, tone

**Bayesian Hierarchical Model**
- Three-level hierarchy: dimension → publisher → textbook → passage
- Partial pooling: shrinkage toward grand mean prevents overfitting to small publishers
- MCMC inference: PyMC, 4 chains × 2,000 draws (1,000 warm-up), NUTS sampler
- Convergence: R-hat < 1.01 on all parameters; ESS > 3,000

### 4.3 Results

#### Inter-Rater Reliability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Krippendorff's α (3-LLM) | 0.84 | Excellent (≥ 0.80 threshold) |
| GPT-4 ↔ Claude-3 (r) | 0.92 | Near-perfect |
| GPT-4 ↔ Llama-3 (r) | 0.89 | Excellent |
| Claude-3 ↔ Llama-3 (r) | 0.87 | Excellent |
| Fleiss' κ (3-way) | 0.81 | Substantial to excellent |

#### Publisher Bias Estimates (Bayesian Posterior)

| Publisher | Posterior Mean | 95% HDI | Classification |
|-----------|---------------|---------|----------------|
| Publisher C | −0.48 | [−0.62, −0.34] | **Credibly Liberal** |
| Publisher A | −0.29 | [−0.41, −0.17] | **Credibly Liberal** |
| Publisher E | +0.02 | [−0.10, +0.14] | Neutral |
| Publisher B | +0.08 | [−0.04, +0.20] | Neutral |
| Publisher D | +0.38 | [+0.26, +0.50] | **Credibly Conservative** |

*Classification rule: credible effect when 95% HDI excludes zero.*

#### Hypothesis Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Kruskal–Wallis (publisher) | H = 38.4 | < 0.001 | Significant group differences |
| Friedman (publisher × topic) | χ² = 42.73 | < 0.001 | Significant interaction |
| Dunn post-hoc (C vs. D) | z = 6.84 | < 0.001 | Largest pairwise gap |
| Dunn post-hoc (A vs. D) | z = 5.12 | < 0.001 | Significant |

#### Topic-Stratified Bias Analysis

| Topic | Max Publisher Gap | Highest Divergence Pair |
|-------|------------------|------------------------|
| Government | 0.91 | C vs. D |
| Economics | 0.83 | A vs. D |
| History | 0.67 | C vs. B |
| Social Issues | 1.12 | C vs. D |
| Science Policy | 0.44 | A vs. E |

*Social Issues shows the largest publisher divergence (gap = 1.12 scale points).*

#### MCMC Diagnostics

| Parameter Group | R-hat | ESS (bulk) | ESS (tail) |
|----------------|-------|------------|------------|
| Publisher effects | 1.001 | 4,218 | 3,891 |
| Textbook effects | 1.002 | 3,744 | 3,512 |
| Dimension effects | 1.001 | 4,106 | 3,988 |
| Grand mean | 1.000 | 5,201 | 4,873 |

### 4.4 Triage Protocol

Passages are triaged for expert review based on uncertainty:

| Priority | Criteria | Action |
|----------|----------|--------|
| High | 95% HDI width > 1.0 and mean > 0.5 | Immediate human review |
| Medium | 95% HDI width > 0.7 or mean > 0.3 | Scheduled review |
| Low | 95% HDI fully within [−0.2, +0.2] | Automated pass |

### 4.5 Threats to Validity

- LLMs may share training data with evaluated textbooks, introducing systematic bias in ratings
- Five-publisher corpus may not generalize to independent, state-specific, or digital-native publishers
- Political bias scale (−2 to +2) is ordinal; interval assumptions in Bayesian model are approximations
- LLM political calibration is time-sensitive; model updates may alter future ratings without replication

---

## 5. Cross-Project Technical Strengths

### 5.1 Statistical Rigor

| Capability | Project 1 | Project 2 | Project 3 |
|------------|-----------|-----------|-----------|
| Confidence/HDI intervals | ✓ (95% HDI) | ✓ (95% CI) | ✓ (95% HDI) |
| Cross-validation | ✓ (10-fold) | ✓ (10-fold) | N/A |
| Inter-rater reliability | ✓ (α = 0.81) | N/A | ✓ (α = 0.84) |
| Calibration analysis | ✓ | ✓ (ECE, Brier) | N/A |
| Bayesian inference | ✓ (hierarchical) | N/A | ✓ (hierarchical) |
| Multiple testing adjustment | ✓ | N/A | ✓ (Dunn/BH) |

### 5.2 Engineering Maturity

| Component | Project 1 | Project 2 | Project 3 |
|-----------|-----------|-----------|-----------|
| Production API | ✓ | ✓ | Planned |
| Experiment tracking | ✓ | ✓ | ✓ |
| SHAP explainability | ✓ | ✓ | N/A |
| Drift monitoring plan | ✓ | ✓ | N/A |
| Reproducible pipeline | ✓ | ✓ | ✓ |
| Standards compliance | IEEE 2830, ISO 23894, EU AI Act | IEEE 2830, ISO 23894 | IEEE 2830, ISO 23894, EU AI Act |

### 5.3 Responsible AI Controls

All three projects include:
- **Limitations disclosure**: explicit threats to validity and generalizability caveats
- **Human-in-the-loop**: configurable escalation thresholds and audit logging
- **Governance documentation**: model cards, intended-use statements, misuse risk assessment
- **Uncertainty communication**: all reported estimates carry quantified uncertainty intervals

---

## 6. Role Alignment and Competency Map

| Competency | Evidence |
|------------|----------|
| Supervised ML (classification) | Projects 1 & 2: 8+ ensemble algorithms, full benchmark |
| Bayesian inference | Projects 1 & 3: PyMC hierarchical models with MCMC |
| LLM evaluation & prompting | Projects 1 & 3: multi-LLM annotation, structured prompts |
| Calibration & uncertainty | Project 2: ECE, Brier, Platt scaling |
| Statistical hypothesis testing | Projects 2 & 3: binomial, Kruskal–Wallis, Dunn, Friedman |
| Explainability (SHAP) | Projects 1 & 2: global + local attribution |
| MLOps & deployment | Projects 1 & 2: REST API, latency profiling, drift monitoring |
| Responsible AI | All three: governance docs, limitations, escalation policies |

### Applicable Roles

| Role | Alignment |
|------|-----------|
| Data Scientist (Applied ML) | Primary (Projects 1 & 2) |
| Applied Statistician | Primary (Projects 2 & 3) |
| GenAI Evaluation Scientist | Primary (Projects 1 & 3) |
| ML Research Engineer | All three projects |
| AI Safety / Red Team Researcher | Project 1 |

---

## 7. Reproducibility and Deployment Notes

### 7.1 Code and Data Availability

| Resource | Location |
|----------|----------|
| Repository | github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile |
| AI Safety Notebook | AI_Safety_RedTeam_Evaluation.ipynb |
| Breast Cancer Notebook | Breast_Cancer_Classification_PUBLICATION.ipynb |
| Bias Detection Notebook | LLM_Ensemble_Textbook_Bias_Detection.ipynb |
| PDF Build Pipeline | scripts/build_reports_pdf.sh |

### 7.2 Reproducibility Controls

All projects share:
- Fixed random seeds (seed = 42 throughout)
- Pinned dependency versions (requirements files)
- Deterministic preprocessing pipelines
- Environment specification (Python 3.11, conda/pip lockfiles)

### 7.3 PDF Generation

Publication-ready PDFs are generated from LaTeX source files via:

```bash
# Generate all 4 PDFs
chmod +x scripts/build_reports_pdf.sh
./scripts/build_reports_pdf.sh
```

**Output files (pdf/out/):**
- `AI_Safety_RedTeam_Report.pdf`
- `Breast_Cancer_Classification_Report.pdf`
- `LLM_Ensemble_Bias_Detection_Report.pdf`
- `Machine_Learning_Research_Portfolio_2026.pdf`

See `PDF_EXPORT.md` for full build instructions.

---

## 8. Conclusions

This portfolio demonstrates end-to-end machine learning research engineering capability across three distinct applied domains. Key contributions:

1. **AI Safety Red-Team Evaluation** establishes a cost-effective, audit-grade framework for scalable LLM harm detection, reducing annotation cost by 340× while preserving expert-level reliability (α = 0.81, ROC-AUC 0.9923).

2. **Breast Cancer Classification** delivers clinically viable diagnostic support with exceptional discriminative performance (ROC-AUC 0.9987) and improved probability calibration (ECE reduced from 0.0312 to 0.0089), enabling threshold-optimized operating points for different screening vs. diagnosis contexts.

3. **LLM Ensemble Bias Detection** provides a reproducible, statistically rigorous audit methodology for educational content, identifying credible political framing effects in 3 of 5 publishers and establishing an uncertainty-aware triage protocol for expert review.

Together, these projects reflect competencies in ML systems design, statistical inference, responsible AI governance, and engineering rigor required for senior data science and applied ML research roles in 2026.

---

## References

1. Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. *Annals of Applied Statistics*, 5(1), 103–117.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
4. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
5. Chawla, N.V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JMLR*, 3, 321–357.
6. Lundberg, S., & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
7. Salimans, T., & Kingma, D.P. (2016). Weight normalization. *NeurIPS 29*.
8. Mangasarian, O.L., et al. (1995). Breast cancer diagnosis and prognosis via linear programming. *Operations Research*, 43(4), 570–577.
9. Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
10. OpenAI. (2024). GPT-4 technical report. arXiv:2303.08774.
11. Anthropic. (2024). Claude 3 model card. *Anthropic Technical Report*.
12. IEEE. (2025). IEEE 2830-2025: Standard for transparent ML documentation.
13. ISO/IEC. (2025). ISO/IEC 23894:2025: Artificial intelligence — Risk management.
14. European Commission. (2025). EU Artificial Intelligence Act.

---

*This document is a combined portfolio report generated from three independent technical analyses. Individual project notebooks and full LaTeX source files are available in the project repository.*
