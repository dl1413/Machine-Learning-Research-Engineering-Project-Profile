# Derek Lankeaux, MS
## Data Scientist | Applied Statistician | LLM Evaluation & GenAI Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

---

### 🎯 Data Scientist (MS, Applied Statistics) | Open to 2026 Full-Time Roles

**Core Competencies:** Experimentation & Causal Inference • Bayesian Statistics • GenAI / LLM Evaluation • Predictive Modeling & MLOps • Stakeholder Communication

> **Data Scientist with an Applied Statistics MS** who turns ambiguous business and research questions into measurable outcomes using experimentation, Bayesian inference, and modern ML. I have shipped end-to-end projects spanning **GenAI evaluation, predictive modeling, and risk analytics** — delivering **96.8–99.12% model performance** with rigorous statistical validation (Krippendorff's α ≥ 0.81, MCMC R-hat < 1.01, p < 0.001). Comfortable owning the full DS workflow: framing the problem, designing the experiment, writing the SQL, building the model, quantifying uncertainty, and communicating impact to non-technical partners. 3 published technical reports aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.

### 🏆 Highlights for 2026 Data Science Roles

- 💸 **340× Cost Reduction via GenAI**: Replaced $6.12 human annotation with $0.018/sample LLM-ensemble pipeline that preserved audit-grade reliability (Krippendorff's α = 0.81) across 12,500 evaluations — a reusable pattern for any team scaling human-in-the-loop labeling
- 🧪 **Causal & Statistical Rigor**: Friedman χ² = 42.73 (p < 0.001), Bayesian hierarchical models with 95% HDI, multiple-testing correction (Bonferroni / FDR), bootstrap CIs, and power analysis — the toolkit DS teams actually use to defend a result
- 🏥 **High-Stakes Predictive Modeling**: 99.12% accuracy, 100% precision, Platt-calibrated probabilities (ECE 0.0089) and threshold tuning for context-specific decision policies (e.g., 100% sensitivity for screening)
- 📊 **Production Data Pipelines**: 80K+ API calls / 2.5M tokens processed with circuit breakers, exponential backoff, and MLflow tracking — repeatable, audited, and monitored
- 🗣️ **Communication & Reproducibility**: 3 publication-grade reports with model cards, calibration plots, and SHAP-based explanations written for both technical reviewers and business stakeholders
- 🧭 **Responsible AI by Default**: IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act-aligned artifacts — increasingly required for DS work touching regulated data (healthcare, finance, education)

---

## 🚀 Featured Research Projects

<table>
<tr>
<td width="50%" valign="top">

### 🛡️ AI Safety Red-Team Evaluation
**[📄 Technical Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md)** | **[📊 Publication](./AI_Safety_RedTeam_Evaluation_Publication.pdf)**

**Automated harm detection using dual-stage LLM ensemble + ML classification**

#### Impact Metrics
- 🎯 **12,500 AI response pairs** evaluated across 6 harm categories
- 📊 **96.8% accuracy** with Stacking Classifier (97.2% precision, 96.1% recall)
- ⚡ **340× cost reduction**: $0.018/sample vs $6.12 human annotation
- 🔬 **Krippendorff's α = 0.81** (excellent LLM ensemble reliability)
- 🚀 **850 samples/hour** processing rate at production scale

#### Technical Innovation
- **Dual-Stage Framework**: LLM ensemble annotation → ML classification pipeline
- **Adversarial Attack Taxonomy**: 8-category MITRE ATLAS-aligned framework; multi-turn escalation (31.8%) identified as highest-risk vector
- **Defense Effectiveness Analysis**: Dual-filter reduces harm rate 21.8% → 4.8% (78% reduction)
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
- **Inter-Publisher Correlation**: Spearman correlation matrix revealing structural editorial relationships (ρ up to 0.74)
- **Cross-Topic Heatmap**: Social Issues shows highest polarization (Δ = 1.36 points across publishers)
- **Passage-Level CIs**: Bootstrap uncertainty quantification flags 12.3% high-uncertainty passages for expert review
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
- **Bayesian Hyperparameter Optimization**: Optuna TPE converges in 5× fewer trials than grid search (45 vs. 240 trials)
- **Model Calibration**: Platt scaling reduces ECE by 71.5% (0.0312 → 0.0089) for clinically reliable confidence
- **Threshold Optimization**: Context-adaptive thresholds (100% sensitivity at 0.31 for mass screening)
- **Advanced Preprocessing**: VIF multicollinearity analysis, SMOTE balancing, RFE feature selection
- **Explainable AI**: SHAP values for clinical transparency, fairness auditing (IEEE 2830-2025)
- **Production Ready**: MLflow registry, FastAPI deployment (<100ms p95 latency)

#### Tech Stack
`scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `Optuna` `SMOTE` `SHAP` `MLflow` `FastAPI`

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

#### 🧪 Experimentation & Causal Inference
- A/B test design & sample size / power analysis
- Frequentist & Bayesian hypothesis testing
- Multiple-testing correction (Bonferroni, FDR)
- Effect sizes (Cohen's d, η², Cramer's V)
- Quasi-experimental & observational designs
- Inter-rater reliability (Krippendorff's α, κ)

**Tools:** SciPy, statsmodels, PyMC, ArviZ, R

</td>
<td width="33%" valign="top">

#### 📊 Modeling & Analytics
- Predictive modeling (classification, regression)
- Ensemble methods (RF, XGBoost, LightGBM, AdaBoost)
- Bayesian hierarchical & probabilistic models
- Feature engineering, selection, calibration
- Uncertainty quantification (HDI, bootstrap)
- SQL & data wrangling at scale

**Tools:** scikit-learn, XGBoost, LightGBM, PyMC, SQL, Pandas, Polars

</td>
<td width="33%" valign="top">

#### 🤖 GenAI & Production DS
- LLM evaluation & benchmarking (GPT-4o, Claude, Llama)
- Prompt engineering & multi-model ensembling
- LLM-as-judge & human-in-the-loop labeling
- Model deployment (FastAPI) & monitoring
- Experiment tracking (MLflow) & reproducibility
- Stakeholder reporting & model cards

**Tools:** OpenAI, Anthropic, HuggingFace, LangChain, MLflow, FastAPI, Docker

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

## 💼 Target Opportunities (2026 Data Science Roles)

### 🎯 Ideal Roles

<table>
<tr>
<td width="50%">

**Data Scientist (Product / Analytics)**
- A/B testing & experimentation platforms
- Causal inference & quasi-experiments
- Metric design & North-Star definition
- SQL-heavy exploratory analysis
- Stakeholder-facing insights & readouts

</td>
<td width="50%">

**Applied / GenAI Data Scientist**
- LLM evaluation & offline benchmarking
- LLM-as-judge & human-in-the-loop pipelines
- Prompt iteration & regression testing
- Retrieval / RAG quality measurement
- Cost & latency optimization

</td>
</tr>
<tr>
<td width="50%">

**Data Scientist (Statistics / Inference)**
- Bayesian hierarchical modeling
- Uncertainty quantification (HDI, calibration)
- Survey design & inter-rater reliability
- Reproducible research workflows
- Statistical consulting for product teams

</td>
<td width="50%">

**Machine Learning / Decision Scientist**
- End-to-end predictive modeling
- Calibration, threshold tuning, decision policies
- Model monitoring & drift detection
- Risk analytics in regulated domains
- Responsible AI / model governance

</td>
</tr>
</table>

### 🌟 What I Bring to a 2026 Data Science Team

✅ **Statistical Foundation**: Applied Statistics MS — Bayesian inference, hypothesis testing, power analysis, causal designs  
✅ **Experimentation Mindset**: A/B test design, multi-arm comparisons, multiple-testing correction, effect-size reporting  
✅ **GenAI Fluency**: Production LLM-ensemble pipelines (GPT-4o, Claude, Llama) with measurable cost-to-quality tradeoffs  
✅ **Modeling Range**: From XGBoost/LightGBM classifiers (99.12% acc) to Bayesian hierarchical models (R-hat < 1.01)  
✅ **Engineering Hygiene**: SQL, Python (Pandas/Polars), MLflow, FastAPI, Docker, Git — ships and maintains pipelines, not just notebooks  
✅ **Communication**: 3 publication-grade reports with model cards, calibration plots, and SHAP-based explanations for non-technical readers  
✅ **Domain Versatility**: Healthcare, education, and AI-safety case studies — comfortable in regulated, high-stakes settings

---

## 📚 Publications & Technical Reports

| Title | Type | Date | Links |
|-------|------|------|-------|
| **AI Safety Red-Team Evaluation** | Technical Report v2.0.0 | Apr 2026 | [Report](./AI%20Safety%20Red-Team%20Evaluation_%20Technical%20Analysis%20Report.md) • [PDF](./AI_Safety_RedTeam_Evaluation_Publication.pdf) |
| **LLM Ensemble Textbook Bias Detection** | Technical Report v4.0.0 | Apr 2026 | [Report](./LLM_Ensemble_Bias_Detection_Report.md) • [PDF](./LLM_Bias_Detection_Publication.pdf) |
| **Breast Cancer Classification** | Technical Report v4.0.0 | Apr 2026 | [Report](./Breast_Cancer_Classification_Report.md) • [PDF](./Breast_Cancer_Classification_Publication.pdf) |

---

## 📫 Let's Connect

<div align="center">

### 🤝 Open to 2026 Data Science Opportunities | Available for Interviews

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

**Data Science:** Experimentation • A/B Testing • Causal Inference • Quasi-Experiments • Metric Design • Cohort Analysis • Funnel Analysis • Power Analysis • Hypothesis Testing • Stakeholder Communication • SQL • Dashboarding

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Bagging • Feature Engineering • Model Calibration • Threshold Tuning

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

**📌 Last Updated:** April 2026  
**✅ Compliance:** IEEE 2830-2025 (Transparent ML) • ISO/IEC 23894:2025 (AI Risk Management)  
**🔒 License:** Portfolio content © 2026 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
