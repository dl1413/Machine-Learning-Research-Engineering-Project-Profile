# Resume Bullet Snippets — Per Role Family

Paste-ready. Each block is the 3-project set, re-tuned to lead with the
signal that role family cares about most. Keep total bullets per project to
3-4 on the resume.

---

## 1. LLM Evaluation / Applied Scientist

**AI Safety Red-Team Evaluation Framework**
- Built dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) reaching 96.8%
  harm-detection accuracy on 12,500 response pairs with Krippendorff's alpha =
  0.81 (audit-grade IRR).
- Cut per-sample cost 340x ($0.018 vs. $6.12 human) at 850 samples/hour;
  Stacking Classifier hit 97.2% precision / 96.1% recall / ROC-AUC 0.9923.
- Quantified per-model risk with a Bayesian hierarchical model (95% HDI,
  R-hat < 1.01) and SHAP-explained every decision under IEEE 2830-2025.

**LLM Ensemble Textbook Bias Detection**
- Scored 67,500 ratings across 4,500 passages (2.5M tokens); 92% pairwise LLM
  correlation, alpha = 0.84.
- Surfaced publisher-level bias with Friedman chi-squared = 42.73, p < 0.001
  via a Bayesian hierarchical model with partial pooling.

**Clinical-Grade Breast Cancer Classification**
- 99.12% accuracy, 100% precision, ROC-AUC 0.9987 — same evaluation rigor
  (calibration, thresholding, SHAP) in a supervised clinical setting.

---

## 2. AI Safety / Responsible AI Engineer

**AI Safety Red-Team Evaluation Framework**
- Dual-stage LLM ensemble across 6 harm categories reaching 96.8% accuracy,
  Krippendorff's alpha = 0.81, at 340x lower cost than human annotation.
- Produced compliance-ready artifacts (SHAP, model card, audit trail) aligned
  to IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.
- Bayesian hierarchical multi-model risk (95% HDI, R-hat < 1.01) for
  cross-model disagreement analysis.

**LLM Ensemble Textbook Bias Detection**
- 67,500 ratings, Bayesian partial pooling, p < 0.001 publisher-level bias
  finding — demonstrates hierarchical evidence on top of LLM judges.

**Clinical-Grade Breast Cancer Classification**
- Explainable ML (SHAP) with a IEEE 2830-2025 model card and fairness audit;
  99.12% accuracy at <100ms p95 FastAPI serving.

---

## 3. GenAI Applied Scientist (Enterprise)

**LLM Ensemble Textbook Bias Detection**
- Processed 67,500 LLM ratings on 4,500 passages (2.5M tokens) through
  GPT-4o / Claude-3.5 / Llama-3.2 with circuit breakers, exponential backoff,
  and MLflow tracking.
- Bayesian hierarchical model with partial pooling and MCMC convergence
  (R-hat < 1.01) → publisher-level 95% HDIs.
- Statistically significant bias in 3/5 publishers (chi-squared = 42.73,
  p < 0.001); 92% pairwise LLM correlation, alpha = 0.84.

**AI Safety Red-Team Evaluation Framework**
- 96.8% accuracy at $0.018/sample (340x cost reduction); Bayesian multi-model
  risk with SHAP explainability and IEEE 2830-2025 audit trail.

**Clinical-Grade Breast Cancer Classification**
- FastAPI/MLflow production stack: 99.12% accuracy, <100ms p95 latency,
  MLflow model registry, SHAP-audited decisions.

---

## 4. Data Scientist (Bayesian / Experimentation)

**LLM Ensemble Textbook Bias Detection**
- Bayesian hierarchical model with partial pooling across 5 publishers /
  4,500 passages; MCMC convergence (R-hat < 1.01, ESS healthy); publisher-level
  95% HDIs.
- Friedman chi-squared = 42.73, p < 0.001 for publisher-level bias;
  Krippendorff's alpha = 0.84, 92% pairwise LLM correlation.
- Reproducible pipeline: circuit breakers, exponential backoff, MLflow.

**AI Safety Red-Team Evaluation Framework**
- IRR at scale: alpha = 0.81 across 12,500 pairs; Bayesian multi-model risk
  with 95% HDI; ROC-AUC 0.9923 with calibration and threshold tuning.

**Clinical-Grade Breast Cancer Classification**
- 8-algorithm benchmark with VIF, SMOTE, RFE, calibration (Platt / isotonic),
  and threshold tuning; 99.12% accuracy, 100% precision.

---

## 5. HealthTech / Clinical ML Engineer

**Clinical-Grade Breast Cancer Classification**
- 99.12% accuracy exceeding human expert performance (90-95%); 100% precision
  (zero false positives), 98.59% recall, ROC-AUC 0.9987.
- Ran 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking,
  Voting) with VIF, SMOTE, RFE, calibration, threshold tuning.
- SHAP-based clinical transparency and fairness audit per IEEE 2830-2025;
  FastAPI serving at <100ms p95 with MLflow model registry.

**AI Safety Red-Team Evaluation Framework**
- Same MLOps discipline in an LLM-safety domain: 96.8% accuracy, 340x cost
  reduction, IEEE 2830-2025 audit trail.

**LLM Ensemble Textbook Bias Detection**
- Bayesian hierarchical modeling with R-hat < 1.01 and p < 0.001 findings —
  evidence of statistical rigor beyond point-estimate ML.
