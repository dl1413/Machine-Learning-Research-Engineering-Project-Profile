# Derek Lankeaux, MS
## Machine Learning Research Engineer | LLM Evaluation Specialist | AI Safety Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

---

### 🎯 Machine Learning Research Engineer | Actively Seeking 2026 Opportunities

**Core Competencies:** Multi-Model LLM Evaluation • Ensemble ML Systems • Bayesian Uncertainty Quantification • Production MLOps • AI Safety Red-Teaming

> **Impact-Driven ML Research Engineer** specialized in building production-grade LLM evaluation frameworks, multi-model ensemble systems, and Bayesian inference pipelines. Proven track record of delivering **96.8-99.12% accuracy** models with rigorous statistical validation. Experienced in processing **80K+ LLM annotations** at production scale (850 samples/hr) while maintaining research-grade reliability (Krippendorff's α ≥ 0.81). Published researcher with **3 technical reports** demonstrating compliance with IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act standards.

### 🏆 Key Achievements for 2026 ML Research Engineer Roles

- 🛡️ **Built LLM Red-Team Framework**: 3-model ensemble (GPT-4o, Claude-3.5, Llama-3.2) with 340× cost reduction ($0.018/sample) and audit-grade reliability (α = 0.81)
- 🔬 **Developed Multi-Model Evaluation Pipeline**: Krippendorff's α = 0.81-0.84 across frontier LLMs with Bayesian uncertainty quantification
- 🏥 **Deployed Clinical-Grade ML System**: 99.12% accuracy breast cancer classifier exceeding human expert performance (90-95%)
- 📊 **Scaled Production NLP Pipelines**: 80K+ API calls with circuit breakers, rate limiting, and MLflow experiment tracking
- ⚡ **Engineered Low-Latency Inference**: FastAPI deployments with <100ms p95 latency and real-time monitoring
- 📈 **Published Research-Quality Reports**: 3 technical publications with p < 0.001 significance, 95% HDI intervals, and SHAP explainability

---

## 🚀 Featured Research Projects

<table>
<tr>
<td width="50%" valign="top">

### 🛡️ AI Safety Red-Team Evaluation
**[📄 Technical Report](./AI_Safety_RedTeam_Evaluation_Report.md)** | **[📊 Publication](./AI_Safety_RedTeam_Evaluation_Publication.pdf)**

**Automated harm detection using dual-stage LLM ensemble + ML classification**

#### Impact Metrics
- 🎯 **12,500 AI response pairs** evaluated across 6 harm categories
- 📊 **96.8% accuracy** with Stacking Classifier (97.2% precision, 96.1% recall)
- ⚡ **340× cost reduction**: $0.018/sample vs $6.12 human annotation
- 🔬 **Krippendorff's α = 0.81** (excellent LLM ensemble reliability)
- 🚀 **850 samples/hour** processing rate at production scale

#### Technical Innovation
- **Dual-Stage Framework**: LLM ensemble annotation → ML classification pipeline
- **Multi-Model Risk Analysis**: Bayesian hierarchical modeling quantifying vulnerability (95% HDI)
- **47 Engineered Features**: Linguistic, semantic, and structural harm signals
- **6 Harm Categories**: Dangerous info, hate, deception, privacy, illegal activity, self-harm
- **Production MLOps**: Scalable deployment with SHAP explainability and audit trails

#### Tech Stack
`GPT-4o` `Claude-3.5` `Llama-3.2` `XGBoost` `Stacking` `PyMC` `SHAP` `MLflow` `Constitutional AI`

</td>
<td width="50%" valign="top">

### 🔬 LLM Ensemble Bias Detection
**[📄 Technical Report](./LLM_Ensemble_Bias_Detection_Report.md)** | **[📊 Publication](./LLM_Bias_Detection_Publication.pdf)**

**Multi-LLM framework for bias detection using Bayesian hierarchical modeling**

#### Impact Metrics
- 📊 **67,500 bias ratings** processed across 4,500 textbook passages
- 🎯 **Krippendorff's α = 0.84** (excellent inter-rater reliability)
- 📈 **Statistically significant findings** (Friedman χ² = 42.73, p < 0.001)
- ⚡ **Production-scale deployment** handling 2.5M tokens

#### Technical Innovation
- **Multi-LLM Ensemble**: GPT-4o, Claude-3.5-Sonnet, Llama-3.2 with 92% pairwise correlation
- **Bayesian Inference**: PyMC hierarchical model with partial pooling, MCMC convergence (R-hat < 1.01)
- **Statistical Rigor**: 95% HDI quantification, publisher-level credible bias detection (3/5 significant)
- **Production Engineering**: Circuit breakers, exponential backoff, MLflow tracking

#### Tech Stack
`GPT-4o` `Claude-3.5` `Llama-3.2` `PyMC` `ArviZ` `MLflow` `FastAPI` `LangChain`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🏥 Breast Cancer ML Classification
**[📄 Technical Report](./Breast_Cancer_Classification_Report.md)** | **[📊 Publication](./Breast_Cancer_Classification_Publication.pdf)**

**Clinical-grade ensemble system exceeding human expert performance**

#### Impact Metrics
- 🏆 **99.12% accuracy** (best-in-class AdaBoost)
- 💯 **100% precision** (zero false positives)
- 🎯 **98.59% recall** (minimal missed cases)
- 📈 **ROC-AUC: 0.9987** (near-perfect discrimination)

#### Technical Innovation
- **8-Algorithm Benchmark**: Comprehensive evaluation (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting)
- **Advanced Preprocessing**: VIF multicollinearity analysis, SMOTE balancing, RFE feature selection
- **Explainable AI**: SHAP values for clinical transparency, fairness auditing (IEEE 2830-2025)
- **Production Ready**: MLflow registry, FastAPI deployment (<100ms p95 latency)

#### Tech Stack
`scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `SMOTE` `SHAP` `MLflow` `FastAPI`

</td>
<td width="50%" valign="top">

### 📊 Research Impact Summary

**Cross-Project Synthesis:**
- **3 production ML systems** deployed across AI safety, bias detection, and healthcare
- **80,000+ annotations** processed via LLM ensembles with validated reliability
- **340× cost efficiency** gain in AI safety evaluation vs human baseline
- **Consistent statistical rigor**: Krippendorff's α ≥ 0.81, MCMC R-hat < 1.01, p < 0.001
- **Reproducible pipelines**: MLflow tracking, versioned artifacts, IEEE 2830-2025 compliance

**Domain Expertise:**
- AI Safety & Red-Teaming
- Educational Content Analysis  
- Clinical Decision Support
- Responsible AI Governance
- Production MLOps at Scale

</td>
</tr>
</table>

---

## 💼 Professional Experience & Capabilities

### 🎯 Core Expertise

<table>
<tr>
<td width="33%" valign="top">

#### 🤖 LLM & AI Safety
- Multi-model ensemble architectures
- AI safety red-team evaluation
- Prompt engineering & optimization
- Inter-rater reliability analysis
- Harm detection & classification
- API integration at scale

**Tools:** GPT-4o, Claude-3.5, Llama-3.2, HuggingFace, LangChain, Constitutional AI

</td>
<td width="33%" valign="top">

#### 📊 Statistical ML
- Ensemble methods (8+ algorithms)
- Bayesian hierarchical modeling
- MCMC diagnostics (R-hat, ESS)
- Hypothesis testing & validation
- Feature engineering & selection

**Tools:** PyMC, ArviZ, scikit-learn, XGBoost, LightGBM

</td>
<td width="33%" valign="top">

#### ⚙️ Production MLOps
- FastAPI model deployment
- MLflow experiment tracking
- Circuit breakers & rate limiting
- Monitoring & drift detection
- Docker/Kubernetes orchestration

**Tools:** MLflow, FastAPI, Docker, Redis, Prometheus

</td>
</tr>
</table>

### 🛠️ Technical Stack

```yaml
Languages:        Python 3.12+ • R • SQL • Bash
ML Frameworks:    PyTorch 2.0+ • TensorFlow 2.15+ • scikit-learn 1.5+ • JAX
LLM APIs:         OpenAI (GPT-4o) • Anthropic (Claude-3.5) • Meta (Llama-3.2) • HuggingFace
Ensemble ML:      XGBoost 2.1+ • LightGBM 4.5+ • CatBoost • AdaBoost
Bayesian Stats:   PyMC 5.15+ • ArviZ 0.18+ • NumPyro • Stan • JAGS
Data Stack:       Pandas 2.2+ • Polars 1.0+ • NumPy 2.0+ • Dask • Apache Arrow
MLOps:            MLflow 2.15+ • Weights & Biases • DVC • Kubeflow
Deployment:       FastAPI 0.110+ • Docker • Kubernetes • AWS • GCP
Monitoring:       Prometheus • Grafana • ELK Stack • Datadog
Explainability:   SHAP • LIME • Captum • InterpretML
Version Control:  Git • GitHub Actions • GitLab CI/CD
```

### 🔬 Research Methodology

**Statistical Rigor**
- ✅ Cross-validation (k-fold, stratified, leave-one-out)
- ✅ Bayesian inference with credible intervals (95% HDI)
- ✅ Multiple testing correction (Bonferroni, FDR, Holm-Sidak)
- ✅ Effect size reporting (Cohen's d, η², Cramer's V)
- ✅ Power analysis and sample size determination

**Reproducibility Standards**
- ✅ IEEE 2830-2025 (Transparent ML) compliance
- ✅ ISO/IEC 23894:2025 (AI Risk Management) alignment
- ✅ Fixed random seeds and version pinning
- ✅ Comprehensive model cards and documentation
- ✅ Carbon footprint tracking and reporting

**Production Engineering**
- ✅ Robust error handling and circuit breakers
- ✅ Adaptive rate limiting and backoff strategies
- ✅ Comprehensive logging (structlog) and monitoring
- ✅ A/B testing frameworks and gradual rollouts
- ✅ Model performance tracking and drift detection

---

## 📊 Quantitative Performance Summary

<table>
<tr>
<td width="33%" valign="top">

### AI Safety Red-Team
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 96.8% | High reliability |
| **Precision** | 97.2% | Low false alarms |
| **Recall** | 96.1% | Comprehensive detection |
| **ROC-AUC** | 0.9923 | Near-perfect |
| **LLM Reliability** | α = 0.81 | Excellent (≥0.80) |
| **Cost Reduction** | 340× | $0.018/sample |
| **Throughput** | 850/hr | Production scale |

</td>
<td width="33%" valign="top">

### LLM Bias Detection
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Inter-Rater Reliability** | α = 0.84 | Excellent (≥0.80) |
| **Model Convergence** | R-hat < 1.01 | Perfect |
| **Statistical Power** | χ² = 42.73 | p < 0.001 |
| **Scale Deployment** | 67.5K ratings | Production |
| **Credible Findings** | 3/5 publishers | 60% detection |

</td>
<td width="33%" valign="top">

### Breast Cancer ML
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 99.12% | Exceeds human (90-95%) |
| **Precision** | 100.00% | Zero false positives |
| **Recall** | 98.59% | Minimal misses |
| **ROC-AUC** | 0.9987 | Near-perfect |
| **CV Stability** | 98.46% ± 1.12% | Robust |

</td>
</tr>
</table>

---

## 🎓 Education & Certifications

**Master of Science in Applied Statistics**  
Rochester Institute of Technology | Expected 2026  
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

**Relevant Coursework:**
- Advanced Bayesian Inference & MCMC Methods
- Deep Learning & Neural Networks
- Statistical Learning Theory
- Experimental Design & Causal Inference
- High-Dimensional Statistics
- Computational Statistics & Optimization

---

## 💼 Target Opportunities (2026 Machine Learning Research Engineer)

### 🎯 Ideal Roles

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

**ML Systems Engineer**
- Scalable inference infrastructure
- MLOps platform development
- Real-time model serving (<100ms)
- Distributed training pipelines
- Monitoring & observability

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

**AI Safety Research Engineer**
- Red-team evaluation frameworks
- Harm detection & classification
- Model alignment research
- Responsible AI governance
- Compliance (IEEE 2830-2025, EU AI Act)

</td>
</tr>
</table>

### 🌟 What I Bring to 2026 ML Research Engineering Roles

✅ **LLM Expertise**: Multi-model ensemble systems (GPT-4o, Claude-3.5, Llama-3.2) with validated reliability (α ≥ 0.81)  
✅ **Production ML**: FastAPI deployments processing 850 samples/hr with <100ms p95 latency  
✅ **Statistical Rigor**: Bayesian inference, hypothesis testing, 95% HDI quantification (p < 0.001)  
✅ **AI Safety**: Pioneered 340× cost-efficient red-team framework with 6-category harm taxonomy  
✅ **Research Quality**: 3 technical reports with IEEE 2830-2025 and EU AI Act compliance  
✅ **MLOps Maturity**: MLflow tracking, model versioning, circuit breakers, monitoring infrastructure  
✅ **Proven Impact**: Clinical-grade ML (99.12% accuracy) exceeding human expert performance

---

## 📚 Publications & Technical Reports

| Title | Type | Date | Links |
|-------|------|------|-------|
| **AI Safety Red-Team Evaluation** | Technical Report v1.0.0 | Jan 2026 | [Report](./AI_Safety_RedTeam_Evaluation_Report.md) • [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| **LLM Ensemble Textbook Bias Detection** | Technical Report v3.0.0 | Jan 2026 | [Report](./LLM_Ensemble_Bias_Detection_Report.md) • [PDF](./LLM_Bias_Detection_Publication.pdf) |
| **Breast Cancer Classification** | Technical Report v3.0.0 | Jan 2026 | [Report](./Breast_Cancer_Classification_Report.md) • [PDF](./Breast_Cancer_Classification_Publication.pdf) |

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

<div align="center">

## 🛠️ Repository Structure

```
Machine-Learning-Research-Engineering-Project-Profile/
├── 📄 README.md                                      # This portfolio
├── 🛡️ AI_Safety_RedTeam_Evaluation_Report.md        # AI safety report
├── 📑 AI_Safety_RedTeam_Evaluation_Publication.pdf  # Publication PDF
├── 📊 Breast_Cancer_Classification_Report.md        # ML technical report
├── 📑 Breast_Cancer_Classification_Publication.pdf  # Publication PDF
├── 🔬 LLM_Ensemble_Bias_Detection_Report.md         # LLM research report
└── 📑 LLM_Bias_Detection_Publication.pdf            # Publication PDF
```

---

### 🔍 Keywords for Search & ATS

</div>

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Bagging • Feature Engineering

**Large Language Models:** GPT-4 • GPT-4o • Claude-3.5-Sonnet • Llama-3.2 • BERT • Transformers • Prompt Engineering • Few-Shot Learning • Zero-Shot Learning • In-Context Learning • Constitutional AI

**AI Safety & Red-Teaming:** Harm Detection • Adversarial Testing • Safety Evaluation • Red Team • Jailbreak Detection • Model Alignment • RLHF • Constitutional AI • Safety Benchmarking • Vulnerability Assessment

**Bayesian Statistics:** Hierarchical Modeling • MCMC • PyMC • Stan • Posterior Inference • Prior Specification • Credible Intervals • Bayesian Inference • Probabilistic Programming • HDI

**Statistical Methods:** Hypothesis Testing • Cross-Validation • Bootstrap • Permutation Testing • Effect Sizes • Power Analysis • Multiple Testing Correction • Inter-Rater Reliability • Krippendorff's Alpha • Cohen's Kappa

**Explainable AI (XAI):** SHAP • LIME • Feature Importance • Model Interpretability • Fairness Auditing • Bias Detection • Responsible AI • AI Ethics • AI Governance • Audit Trails

**MLOps & Production:** MLflow • Weights & Biases • Model Registry • Experiment Tracking • FastAPI • Docker • Kubernetes • CI/CD • Model Monitoring • Drift Detection • A/B Testing • Circuit Breakers

**Programming:** Python • R • SQL • PyTorch • TensorFlow • scikit-learn • Pandas • NumPy • Dask • Apache Spark

**Research Engineering:** Technical Writing • Statistical Validation • Reproducible Research • Peer Review • Literature Review • Experimental Design • Causal Inference • Cost-Benefit Analysis

**AI Safety Domains:** Dangerous Information • Hate Speech • Deception Detection • Privacy Violation • Illegal Activity • Self-Harm Prevention • Content Moderation • Trust & Safety

**Standards & Compliance:** IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act • GDPR • Model Cards • Transparency • Accountability • AI Governance

---

<div align="center">

**📌 Last Updated:** January 2026  
**✅ Compliance:** IEEE 2830-2025 (Transparent ML) • ISO/IEC 23894:2025 (AI Risk Management)  
**🔒 License:** Portfolio content © 2026 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
