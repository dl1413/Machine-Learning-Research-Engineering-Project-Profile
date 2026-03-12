# Derek Lankeaux, MS
## Machine Learning Research Engineer | LLM Evaluation Specialist | AI Safety Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

---

## 🎯 Machine Learning Research Engineer | Actively Seeking 2026 Opportunities

**Core Competencies:** Multi-Model LLM Evaluation • Ensemble ML Systems • Bayesian Uncertainty Quantification • Production MLOps • AI Safety Red-Teaming

> **Impact-Driven ML Research Engineer** specialized in building production-grade LLM evaluation frameworks, multi-model ensemble systems, and Bayesian inference pipelines. Proven track record delivering **96.8-99.12% accuracy** models with rigorous statistical validation. Experienced processing **80K+ LLM annotations** at production scale (850 samples/hr) while maintaining research-grade reliability (Krippendorff's α ≥ 0.81).

---

## 🚀 Integrated Research Portfolio: 3-Part ML Engineering Showcase

This portfolio demonstrates comprehensive machine learning research engineering capabilities through three interconnected projects spanning **AI Safety**, **NLP/LLM Evaluation**, and **Clinical ML**—showcasing the full ML lifecycle from research to production deployment.

<table>
<tr>
<td colspan="3" align="center">
<h3>📊 Portfolio-Wide Impact Metrics</h3>
</td>
</tr>
<tr>
<td width="33%" valign="top">

**Scale & Efficiency**
- 80K+ LLM API calls processed
- 340× cost reduction in AI safety
- 850 samples/hr throughput
- <100ms p95 latency

</td>
<td width="33%" valign="top">

**Statistical Rigor**
- Krippendorff's α: 0.81-0.84
- 95% HDI quantification
- MCMC R-hat < 1.01
- p < 0.001 significance

</td>
<td width="33%" valign="top">

**Model Performance**
- 96.8-99.12% accuracy
- 97.2-100% precision
- ROC-AUC: 0.9923-0.9987
- Clinical-grade validation

</td>
</tr>
</table>

---

## 🔬 Part 1: AI Safety Red-Team Evaluation

**[📄 Full Technical Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report%20(3).md)** | **[📊 Publication](./AI_Safety_RedTeam_Evaluation_Publication%20(1).pdf)**

**Dual-Stage LLM Ensemble + ML Classification for Automated Harm Detection**

### Executive Summary
Built production-ready AI safety evaluation framework combining 3 frontier LLMs (GPT-4o, Claude-3.5, Llama-3.2) for annotation with ensemble ML classification. Achieved **96.8% accuracy** detecting harmful AI outputs across 6 harm categories, processing **12,500 prompt-response pairs** with excellent inter-rater reliability (α = 0.81).

### Key Innovations
- **340× Cost Reduction:** $0.018/sample vs $6.12 human annotation
- **Production Scale:** 850 samples/hr processing rate
- **Bayesian Risk Modeling:** Quantified model vulnerability with 95% HDI
- **6-Category Taxonomy:** Dangerous info, hate, deception, privacy, illegal, self-harm
- **Audit-Grade Reliability:** α = 0.81 inter-rater agreement

### Technical Highlights
```
LLM Ensemble → 47 Engineered Features → Stacking Classifier → SHAP Explainability
                   ↓
            Krippendorff's α = 0.81
                   ↓
        96.8% Accuracy, 97.2% Precision, 96.1% Recall
```

<details>
<summary><b>Performance Metrics</b></summary>

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 96.8% | Target: >95% |
| **Precision** | 97.2% | Industry-leading |
| **Recall** | 96.1% | Critical for safety |
| **ROC-AUC** | 0.9923 | Near-perfect |
| **Krippendorff's α** | 0.81 | Excellent (≥0.80) |
| **Processing Rate** | 850/hr | 100× human baseline |
| **Cost** | $0.018/sample | 340× reduction |

</details>

**Tech Stack:** `GPT-4o` `Claude-3.5` `Llama-3.2` `XGBoost` `Stacking` `PyMC` `SHAP` `MLflow` `Constitutional AI`

---

## 📚 Part 2: LLM Ensemble Bias Detection

**[📄 Full Technical Report](./LLM_Ensemble_Bias_Detection_Report%20(3).md)** | **[📊 Publication](./LLM_Bias_Detection_Publication%20(6).pdf)**

**Multi-LLM Framework with Bayesian Hierarchical Modeling for Educational Content Analysis**

### Executive Summary
Developed scalable bias detection framework using 3-model LLM ensemble to analyze **4,500 textbook passages** across 5 publishers, generating **67,500 bias ratings**. Bayesian hierarchical modeling quantified publisher-level effects with 95% credible intervals, achieving **α = 0.84** inter-rater reliability.

### Key Innovations
- **Statistically Credible Findings:** 3/5 publishers showed significant bias (p < 0.001)
- **Excellent Reliability:** α = 0.84 LLM ensemble agreement
- **Full Uncertainty Quantification:** Bayesian posteriors with 95% HDI
- **Production NLP:** 2.5M tokens processed with circuit breakers
- **Hierarchical Modeling:** Partial pooling across publishers/textbooks

### Technical Highlights
```
GPT-4o + Claude-3.5 + Llama-3.2 → Ensemble Mean → PyMC Hierarchical Model
        ↓                              ↓                     ↓
   67,500 ratings              α = 0.84         Friedman χ² = 42.73 (p<0.001)
```

<details>
<summary><b>Performance Metrics</b></summary>

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Krippendorff's α** | 0.84 | Excellent reliability |
| **Pairwise r (GPT↔Claude)** | 0.92 | Near-perfect agreement |
| **Statistical Power** | χ² = 42.73 | p < 0.001 |
| **MCMC Convergence** | R-hat < 1.01 | Perfect |
| **Scale Deployment** | 67.5K ratings | Production-ready |
| **ESS** | >3,000 | Adequate sampling |
| **Credible Findings** | 3/5 publishers | 60% detection rate |

</details>

**Tech Stack:** `GPT-4o` `Claude-3.5` `Llama-3.2` `PyMC` `ArviZ` `MLflow` `FastAPI` `LangChain`

---

## 🏥 Part 3: Breast Cancer ML Classification

**[📄 Full Technical Report](./Breast_Cancer_Classification_Report%20(4).md)** | **[📊 Publication](./Breast_Cancer_Classification_Publication%20(6).pdf)**

**Clinical-Grade Ensemble System Exceeding Human Expert Performance**

### Executive Summary
Implemented comprehensive ML pipeline comparing 8 ensemble algorithms for breast cancer classification. Best model (AdaBoost) achieved **99.12% accuracy**, **100% precision**, **98.59% recall**—exceeding human inter-observer agreement (90-95%) on Wisconsin Diagnostic Breast Cancer dataset.

### Key Innovations
- **Clinical-Grade Performance:** 99.12% accuracy vs 90-95% human experts
- **Zero False Positives:** 100% precision on test set
- **Comprehensive Benchmarking:** 8 ensemble algorithms evaluated
- **Feature Engineering:** VIF analysis, SMOTE balancing, RFE selection
- **Production Ready:** MLflow tracking, FastAPI deployment (<100ms)

### Technical Highlights
```
WDBC Dataset → VIF Analysis → SMOTE → RFE → 8 Ensemble Models → SHAP
    569 samples     ↓        1:1 ratio  15 feat    ↓            ↓
                Multicollinearity            AdaBoost    Explainability
                   Removal                  99.12% Acc
```

<details>
<summary><b>Performance Metrics</b></summary>

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Accuracy** | 99.12% | Exceptional diagnostic performance |
| **Precision** | 100.00% | Zero false positives |
| **Recall** | 98.59% | Minimal missed malignancies |
| **Specificity** | 100.00% | Perfect benign identification |
| **ROC-AUC** | 0.9987 | Near-perfect discrimination |
| **Cross-Val** | 98.46% ± 1.12% | Robust generalization |
| **Cohen's κ** | 0.9823 | Almost perfect agreement |

</details>

**Tech Stack:** `scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `SMOTE` `SHAP` `MLflow` `FastAPI`

---

## 💼 Integrated Skills Demonstrated Across Portfolio

<table>
<tr>
<td width="33%" valign="top">

### 🤖 LLM & AI Safety
- Multi-model ensemble architectures
- Constitutional AI evaluation
- Red-team methodology
- Inter-rater reliability (α ≥ 0.80)
- Prompt engineering at scale
- Harm detection & classification

**Tools:** GPT-4o, Claude-3.5, Llama-3.2, HuggingFace, LangChain

</td>
<td width="33%" valign="top">

### 📊 Statistical ML
- Ensemble methods (8+ algorithms)
- Bayesian hierarchical modeling
- MCMC diagnostics (R-hat, ESS)
- Krippendorff's α reliability
- Feature engineering (47 features)
- Cross-validation & hypothesis testing

**Tools:** PyMC, ArviZ, scikit-learn, XGBoost, LightGBM, AdaBoost

</td>
<td width="33%" valign="top">

### ⚙️ Production MLOps
- FastAPI model serving
- MLflow experiment tracking
- Circuit breakers & rate limiting
- <100ms p95 latency
- SHAP explainability
- IEEE 2830-2025 compliance

**Tools:** MLflow, FastAPI, Docker, SHAP, Prometheus

</td>
</tr>
</table>

---

## 🛠️ Complete Technical Stack

```yaml
Languages:        Python 3.12+ • R • SQL • Bash
ML Frameworks:    PyTorch 2.0+ • TensorFlow 2.15+ • scikit-learn 1.5+ • JAX
LLM APIs:         OpenAI (GPT-4o) • Anthropic (Claude-3.5) • Meta (Llama-3.2) • HuggingFace
Ensemble ML:      XGBoost 2.1+ • LightGBM 4.5+ • AdaBoost • Stacking • Voting
Bayesian Stats:   PyMC 5.15+ • ArviZ 0.18+ • NumPyro • Stan • MCMC
Data Stack:       Pandas 2.2+ • Polars 1.0+ • NumPy 2.0+ • Apache Arrow
MLOps:            MLflow 2.15+ • Weights & Biases • FastAPI 0.110+ • Docker
Explainability:   SHAP • LIME • InterpretML • Model Cards
Standards:        IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act
```

---

## 📈 Cross-Project Synthesis

### Research Methodology Excellence
- ✅ Bayesian inference with 95% HDI credible intervals
- ✅ Krippendorff's α = 0.81-0.84 (excellent inter-rater reliability)
- ✅ Multiple testing correction (Bonferroni, FDR)
- ✅ MCMC convergence diagnostics (R-hat < 1.01)
- ✅ Bootstrap confidence intervals
- ✅ Statistical power analysis

### Production Engineering Standards
- ✅ Circuit breakers & exponential backoff
- ✅ Rate limiting & adaptive retry logic
- ✅ Comprehensive error handling & logging
- ✅ MLflow experiment tracking & model registry
- ✅ SHAP explainability & audit trails
- ✅ FastAPI deployment (<100ms latency)

### Reproducibility & Compliance
- ✅ IEEE 2830-2025 (Transparent ML) compliance
- ✅ ISO/IEC 23894:2025 (AI Risk Management) alignment
- ✅ EU AI Act transparency requirements
- ✅ Fixed random seeds & version pinning
- ✅ Model cards & documentation
- ✅ Carbon footprint tracking

---

## 🎯 Target 2026 Roles

<table>
<tr>
<td width="50%">

**Machine Learning Research Engineer**
- LLM evaluation & benchmarking systems
- Multi-model ensemble architectures
- Foundation model red-teaming
- Production ML pipeline development
- Model performance optimization

</td>
<td width="50%">

**AI Safety Research Engineer**
- Red-team evaluation frameworks
- Harm detection & classification
- Model alignment research
- Responsible AI governance
- IEEE 2830-2025 compliance

</td>
</tr>
<tr>
<td width="50%">

**Applied Research Scientist**
- Bayesian uncertainty quantification
- Ensemble learning methodologies
- Statistical validation frameworks
- Experimental design & analysis
- Publication-ready research

</td>
<td width="50%">

**ML Systems Engineer**
- Scalable inference infrastructure
- MLOps platform development
- Real-time model serving (<100ms)
- Distributed training pipelines
- Monitoring & observability

</td>
</tr>
</table>

---

## 🏆 Portfolio Impact Summary

| Domain | Scale | Accuracy | Innovation |
|--------|-------|----------|------------|
| **AI Safety** | 12.5K samples | 96.8% | 340× cost reduction |
| **Bias Detection** | 67.5K ratings | α = 0.84 | Bayesian hierarchical modeling |
| **Clinical ML** | 569 patients | 99.12% | Exceeds human performance |
| **Aggregate** | **80K+ annotations** | **96.8-99.12%** | **Production-ready pipelines** |

---

## 📚 Publications

| Title | Type | Date | Links |
|-------|------|------|-------|
| **AI Safety Red-Team Evaluation** | Technical Report v1.0.0 | Jan 2026 | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report%20(3).md) • [PDF](./AI_Safety_RedTeam_Evaluation_Publication%20(1).pdf) |
| **LLM Ensemble Textbook Bias Detection** | Technical Report v3.0.0 | Jan 2026 | [Report](./LLM_Ensemble_Bias_Detection_Report%20(3).md) • [PDF](./LLM_Bias_Detection_Publication%20(6).pdf) |
| **Breast Cancer Classification** | Technical Report v3.0.0 | Jan 2026 | [Report](./Breast_Cancer_Classification_Report%20(4).md) • [PDF](./Breast_Cancer_Classification_Publication%20(6).pdf) |

---

## 🎓 Education

**Master of Science in Applied Statistics**
Rochester Institute of Technology | Expected 2026
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

---

## 📫 Let's Connect

<div align="center">

### 🤝 Open to Research Engineer Opportunities | Available for Interviews

**Preferred Contact:** [LinkedIn](https://linkedin.com/in/derek-lankeaux) • Email Available Upon Request

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Derek_Lankeaux-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-@dl1413-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

**Location:** Available for remote/hybrid positions
**Timeline:** Seeking positions starting 2026
**Visa Status:** Authorized to work in the United States

</div>

---

## 🔍 Keywords for ATS & Search

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Feature Engineering

**Large Language Models:** GPT-4o • Claude-3.5-Sonnet • Llama-3.2 • BERT • Transformers • Prompt Engineering • Constitutional AI • LLM Evaluation

**AI Safety:** Harm Detection • Red-Team • Adversarial Testing • Model Alignment • RLHF • Safety Benchmarking • Vulnerability Assessment

**Bayesian Statistics:** Hierarchical Modeling • MCMC • PyMC • Posterior Inference • Credible Intervals • HDI

**Statistical Methods:** Hypothesis Testing • Cross-Validation • Krippendorff's Alpha • Inter-Rater Reliability • Effect Sizes • Multiple Testing Correction

**Explainable AI:** SHAP • LIME • Feature Importance • Model Interpretability • Fairness Auditing • Responsible AI

**MLOps:** MLflow • Model Registry • Experiment Tracking • FastAPI • Docker • CI/CD • Monitoring • Drift Detection • Circuit Breakers

**Standards:** IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act • Model Cards • AI Governance

---

<div align="center">

**📌 Last Updated:** January 2026
**✅ Compliance:** IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act
**🔒 License:** Portfolio content © 2026 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
