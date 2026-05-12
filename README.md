# Derek Lankeaux, MS
## Machine Learning Research Engineer | LLM Evaluation Specialist | AI Safety Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

---

### 🎯 Machine Learning Research Engineer | Actively Seeking 2026 Opportunities

**Core Competencies:** Multi-Model LLM Evaluation • Ensemble ML Systems • Bayesian Uncertainty Quantification • Production MLOps • AI Safety Red-Teaming

> **Impact-Driven ML Research Engineer** specialized in frontier-LLM evaluation, multi-model ensemble systems, and Bayesian inference pipelines. Currently evaluating frontier models (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, DeepSeek-R1) in production at Toloka AI and developing LLM-powered features at Handshake (20M+ student platform). Proven track record of delivering **96.8-99.12% accuracy** models with rigorous statistical validation. Experienced in processing **10K+ daily inferences** at production scale while maintaining research-grade reliability (Krippendorff's α ≥ 0.81). Published researcher with **3 technical reports** demonstrating compliance with IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act standards. Expertise in Bayesian hyperparameter optimization (Optuna TPE), model calibration, adversarial attack taxonomy, and multi-publisher correlation analysis.

### 🏆 Key Achievements for 2026 ML Research Engineer Roles

- 🛡️ **Built LLM Red-Team Framework**: 3-model ensemble (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro) processing 12,500 adversarial pairs across 12 attack categories; 340× cost reduction ($0.018/sample), α = 0.81 reliability, 5,800+ dimensional feature space
- 🔬 **Developed Multi-Model Evaluation Pipeline**: Krippendorff's α = 0.81-0.84 across frontier LLMs with Bayesian uncertainty quantification; 67,500 ratings over 150 textbooks, systematic bias identified in 60% of publishers
- 🏥 **Deployed Clinical-Grade ML System**: 99.12% accuracy breast cancer classifier exceeding human expert performance (90-95%) with Optuna-optimized hyperparameters and Platt-calibrated probabilities (ECE: 0.0089)
- 🛠️ **Scaling Production LLM Workflows**: Currently 10,000+ daily inferences at Toloka AI; LLM-powered features for 20M+ students at Handshake
- ⚡ **Engineered Low-Latency Inference**: FastAPI deployments with <100ms p95 latency and real-time monitoring
- 📈 **Published Research-Quality Reports**: 3 technical publications with p < 0.001 significance, 95% HDI intervals, and SHAP explainability

---

## 🚀 Featured Research Projects

<table>
<tr>
<td width="50%" valign="top">

### 🛡️ AI Safety Red-Team Evaluation
**[📄 Technical Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md)** | **[📊 Publication](./AI_Safety_RedTeam_Evaluation_Publication.pdf)**

**Automated harm detection using dual-stage LLM ensemble + ML classification**
*Oct 2025 – Dec 2025*

#### Impact Metrics
- 🎯 **12,500 adversarial pairs** evaluated across 12 attack categories
- 📊 **96.8% accuracy** with Stacking Classifier (97.2% precision, 96.1% recall)
- ⚡ **340× cost reduction**: $0.018/sample vs $6.12 human annotation
- 🔬 **Krippendorff's α = 0.81** (excellent LLM ensemble reliability)
- 🚀 **850 samples/hour** processing rate at production scale

#### Technical Innovation
- **Dual-Stage Framework**: LLM ensemble annotation → ML classification pipeline
- **Adversarial Attack Taxonomy**: MITRE ATLAS-aligned framework across 12 attack categories
- **Defense Effectiveness Analysis**: Dual-filter reduces harm rate 21.8% → 4.8% (78% reduction)
- **Multi-Model Risk Analysis**: Bayesian hierarchical modeling quantifying vulnerability (95% HDI)
- **5,800+ Dimensional Feature Space**: Linguistic, semantic, and structural harm signals
- **Production MLOps**: Continuous monitoring (KS test for drift), SHAP explainability, audit trails per IEEE 2830-2025

#### Tech Stack
`GPT-5.1` `Claude Opus 4.5` `Gemini 3 Pro` `Llama 4` `XGBoost` `Stacking` `PyMC` `SHAP` `MLflow` `Constitutional AI`

</td>
<td width="50%" valign="top">

### 🔬 LLM Ensemble Bias Detection
**[📄 Technical Report](./LLM_Ensemble_Bias_Detection_Report.md)** | **[📊 Publication](./LLM_Bias_Detection_Publication.pdf)**

**Multi-LLM framework for bias quantification using Bayesian hierarchical modeling**
*Jun 2025 – Sep 2025*

#### Impact Metrics
- 📚 **150 textbooks** analyzed across publishers
- 📊 **67,500 bias ratings** processed across 4,500 passages
- 🎯 **Krippendorff's α = 0.84** with pairwise correlations r = 0.87–0.92
- 📈 **Systematic bias identified in 60% of publishers** (Friedman χ² = 42.73, p < 0.001)

#### Technical Innovation
- **Multi-LLM Ensemble**: GPT-5.1, Claude Opus 4.5, Gemini 3 Pro with 87–92% pairwise correlation
- **Bayesian Inference**: PyMC hierarchical model with partial pooling, MCMC convergence (R-hat < 1.01, ESS > 3,000)
- **Statistical Rigor**: 95% HDI quantification, publisher-level credible bias detection (60% of publishers)
- **Inter-Publisher Correlation**: Spearman correlation matrix revealing structural editorial relationships (ρ up to 0.74)
- **Cross-Topic Heatmap**: Social Issues shows highest polarization (Δ = 1.36 points across publishers)
- **Passage-Level CIs**: Bootstrap uncertainty quantification flags 12.3% high-uncertainty passages for expert review
- **Production Engineering**: Circuit breakers, exponential backoff, MLflow tracking

#### Tech Stack
`GPT-5.1` `Claude Opus 4.5` `Gemini 3 Pro` `PyMC` `ArviZ` `MLflow` `FastAPI` `LangChain`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🏥 Breast Cancer ML Classification
**[📄 Technical Report](./Breast_Cancer_Classification_Report.md)** | **[📊 Publication](./Breast_Cancer_Classification_Publication.pdf)**

**Clinical-grade ensemble system exceeding human expert performance**
*May 2023 – Dec 2023*

#### Impact Metrics
- 🏆 **99.12% accuracy** (best-in-class AdaBoost)
- 💯 **100% precision** (zero false positives)
- 🎯 **98.59% recall** (minimal missed cases)
- 📈 **ROC-AUC: 0.9987** (near-perfect discrimination)

#### Technical Innovation
- **8-Algorithm Benchmark**: Comprehensive evaluation (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting)
- **Bayesian Hyperparameter Optimization**: Optuna TPE converges in 5× fewer trials than grid search (45 vs. 240 trials)
- **Model Calibration**: Platt scaling reduces ECE by 71.5% (0.0312 → 0.0089) for clinically reliable confidence
- **Threshold Optimization**: Context-adaptive thresholds (100% sensitivity at 0.31 for mass screening)
- **Advanced Preprocessing**: VIF multicollinearity analysis, SMOTE balancing, RFE feature selection
- **Explainable AI**: SHAP + Fairlearn for clinical transparency and fairness auditing (IEEE 2830-2025)
- **Production Ready**: MLflow registry, FastAPI deployment (<100ms p95 latency)

#### Tech Stack
`scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `Optuna` `SMOTE` `SHAP` `Fairlearn` `MLflow` `FastAPI`

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

### 🏢 Work Experience

**AI Quality Assurance Specialist** — Toloka AI (Mindrift) · *Oct 2025 – Present*
- Evaluate frontier LLMs (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4) processing **10,000+ daily inferences** across reasoning, instruction-following, and factual accuracy
- Implement harm detection achieving **<5% false negative rate** across 6 categories
- Design Constitutional AI compliance metrics and autonomous AI agent behavior assessments; annotations integrated into fine-tuning pipelines with MLflow tracking

**AI Trainer** — Handshake · *Nov 2025 – Present*
- Develop LLM-powered features for career platform serving **20M+ students**
- Conduct evaluation of frontier models (GPT-5.1, Claude Opus 4.5) for production deployment using LLM-as-Judge methodologies
- Design A/B testing frameworks and analyze engagement metrics for ML-driven features

**Cloud Software Analyst** — BRdata Software Solutions · *Apr 2024 – Nov 2024*
- Developed NL2SQL training dataset using LangChain + Groq API (chain-of-thought reasoning, few-shot learning); **180 samples/hour** with Redis caching
- Engineered backend pipelines integrating SQL Server with cloud architecture — **10% latency reduction**, **20% reliability improvement** via Docker orchestration

**Mathematics Instructor** — Huntington Learning Center · *Oct 2021 – Dec 2023*
- Delivered data-driven instruction to 30+ students; raised standardized test scores up to **150 points** through curriculum interventions; maintained 95%+ satisfaction

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

**Tools:** GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4, DeepSeek-R1, HuggingFace, LangChain, Constitutional AI

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
LLM APIs:         OpenAI (GPT-5.1) • Anthropic (Claude Opus 4.5) • Google (Gemini 3 Pro) • Meta (Llama 4) • DeepSeek-R1 • HuggingFace
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
| **Credible Findings** | 60% of publishers | Systematic bias |

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
Rochester Institute of Technology *(in progress)*  
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

**Bachelor of Science in Applied Mathematics & Statistics**  
Stony Brook University

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

✅ **LLM Expertise**: Multi-model ensemble systems (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Llama 4) with validated reliability (α ≥ 0.81)  
✅ **Production ML**: Currently 10,000+ daily inferences at Toloka AI; FastAPI deployments with <100ms p95 latency  
✅ **Statistical Rigor**: Bayesian inference, hypothesis testing, 95% HDI quantification (p < 0.001)  
✅ **AI Safety**: Pioneered 340× cost-efficient red-team framework spanning 12 attack categories  
✅ **Research Quality**: 3 technical reports with IEEE 2830-2025 and EU AI Act compliance  
✅ **MLOps Maturity**: MLflow tracking, model versioning, circuit breakers, monitoring infrastructure  
✅ **Proven Impact**: Clinical-grade ML (99.12% accuracy) exceeding human expert performance

---

## 📚 Publications & Technical Reports

| Title | Type | Date | Links |
|-------|------|------|-------|
| **AI Safety Red-Team Evaluation** | Technical Report | Project: Oct–Dec 2025 | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) • [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| **LLM Ensemble Textbook Bias Quantification** | Technical Report | Project: Jun–Sep 2025 | [Report](./LLM_Ensemble_Bias_Detection_Report.md) • [PDF](./LLM_Bias_Detection_Publication.pdf) |
| **Breast Cancer Classification** | Technical Report | Project: May–Dec 2023 | [Report](./Breast_Cancer_Classification_Report.md) • [PDF](./Breast_Cancer_Classification_Publication.pdf) |

---

## 📫 Let's Connect

<div align="center">

### 🤝 Open to Research Engineer Opportunities | Available for Interviews

**Preferred Contact:** [LinkedIn](https://linkedin.com/in/derek-lankeaux) • Email Available Upon Request

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Derek_Lankeaux-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-@dl1413-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

**Location:** Greenlawn, NY (Long Island) · Open to NYC, hybrid, or remote  
**Contact:** (631) 626-0511 · dl1413@g.rit.edu  
**Visa Status:** Authorized to work in the United States

</div>

---

<div align="center">

## 🛠️ Repository Structure

```
LLM-Portfolio/
├── 📄 README.md                                           # This portfolio
├── 🌐 index.html                                          # Interactive portfolio site
├── 🎨 styles.css                                          # Portfolio styling
├── 🛡️ AI Safety Red-Team Evaluation_ Technical...md       # AI safety report
├── 📑 AI_Safety_RedTeam_Evaluation_Publication.pdf             # Publication PDF
├── 📊 Breast_Cancer_Classification_Report.md              # ML technical report
├── 📑 Breast_Cancer_Classification_Publication.pdf        # Publication PDF
├── 🔬 LLM_Ensemble_Bias_Detection_Report.md               # LLM research report
├── 📑 LLM_Bias_Detection_Publication.pdf                  # Publication PDF
└── 📁 reports/                                            # Additional documentation
```

---

### 🔍 Keywords for Search & ATS

</div>

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Bagging • Feature Engineering

**Large Language Models:** GPT-5.1 • Claude Opus 4.5 • Gemini 3 Pro • Llama 4 • DeepSeek-R1 • BERT • Transformers • Prompt Engineering • Few-Shot Learning • Zero-Shot Learning • In-Context Learning • Constitutional AI • LLM-as-Judge

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

**📌 Last Updated:** April 2026  
**✅ Compliance:** IEEE 2830-2025 (Transparent ML) • ISO/IEC 23894:2025 (AI Risk Management)  
**🔒 License:** Portfolio content © 2026 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
