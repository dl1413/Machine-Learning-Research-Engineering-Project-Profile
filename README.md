# Derek Lankeaux, MS
## Data Scientist | Applied Statistician | LLM Evaluation & GenAI Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

---

### 🎯 Data Scientist (MS, Applied Statistics) | Open to 2026 Full-Time Roles

**Core Competencies:** Experimentation & Causal Inference • Bayesian Statistics • GenAI / LLM Evaluation • Predictive Modeling & MLOps • Stakeholder Communication

> **Data Scientist with an Applied Statistics MS** who turns ambiguous business and research questions into measurable outcomes using experimentation, Bayesian inference, and modern ML. Two end-to-end portfolio projects — **LLM-ensemble bias evaluation with Bayesian hierarchical inference** (67.5K ratings, α = 0.84, p < 0.001) and a **calibrated binary classifier with decision-policy tuning** (99.12% accuracy, ECE = 0.0089, two operating points tied to cost ratios) — each shipped with model cards, posterior plots, and SHAP attributions. Comfortable owning the full DS workflow: framing the problem, designing the experiment, writing the SQL, building the model, quantifying uncertainty, and communicating impact to non-technical partners. Both reports align with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.

### 🏆 Highlights for 2026 Data Science Roles

- 🧪 **Statistical Rigor Built In**: Bayesian hierarchical models (PyMC, R-hat < 1.01), Friedman χ² = 42.73 (p < 0.001), 95% HDIs, multiple-testing correction (Bonferroni / FDR), bootstrap CIs, and inter-rater reliability (Krippendorff's α = 0.84) — the toolkit DS teams use to defend a finding to a non-technical stakeholder
- 🏥 **High-Stakes Predictive Modeling**: 99.12% accuracy, 100% precision, Platt-calibrated probabilities (ECE 0.0089), and context-specific threshold tuning (100% sensitivity for mass screening, 100% precision for confirmation) — calibration discipline most accuracy-chasing pipelines skip
- 🔬 **GenAI Evaluation Pipelines**: LLM-as-judge ensembles across GPT-4o, Claude-3.5, and Llama-3.2 with 92% pairwise correlation — a reusable pattern for scaling labeling, A/B-testing prompts, and quantifying disagreement as uncertainty
- 📊 **Production Data Pipelines**: 67.5K LLM ratings / 2.5M tokens processed with circuit breakers, exponential backoff, deterministic re-runs, and MLflow tracking — repeatable, audited, monitored
- 🗣️ **Communication & Reproducibility**: 2 publication-grade reports with model cards, calibration plots, SHAP attributions, and posterior visualizations written for both technical reviewers and business stakeholders
- 🧭 **Responsible AI by Default**: IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act-aligned artifacts — table stakes for DS work in regulated domains (healthcare, finance, education)

---

## 🚀 Featured Data Science Projects

Two end-to-end case studies — one in **GenAI evaluation with Bayesian inference**, one in **high-stakes predictive modeling with calibrated decision policies**. Each one walks through the problem framing, methods, results, and what I'd actually ship to a stakeholder.

---

### 🔬 Project 1 — LLM Ensemble Textbook Bias Detection
**[📄 Technical Report](./LLM_Ensemble_Bias_Detection_Report.md)** | **[📊 Publication PDF](./LLM_Bias_Detection_Publication.pdf)**

> *How do you measure a subjective construct (political bias in textbooks) at scale, with quantified uncertainty, and produce a finding a school district or publisher would actually defend in front of a school board?*

#### The problem
Educational content shapes how millions of students learn history, civics, and economics. Stakeholders — districts, parents, publishers — want to know whether textbook content leans systematically in one direction. The traditional answer (expert human reviewers) is slow, expensive, and statistically thin: small N, no inter-rater agreement reporting, no uncertainty quantification on the headline number. The DS question: can we replace or augment expert review with an LLM-ensemble pipeline that is fast, cheap, *and* statistically defensible?

#### The approach
**Stage 1 — Multi-rater LLM annotation.** Three frontier LLMs (GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B) independently rated 4,500 passages from 150 textbooks on a [-2, +2] bias scale, generating **67,500 ratings** across 2.5M tokens. Prompts were engineered with rubrics, anchoring examples, and chain-of-thought elicitation; an LLM-as-judge setup with deliberate disagreement (rather than single-model labeling) is what unlocks uncertainty quantification downstream.

**Stage 2 — Inter-rater reliability validation.** Krippendorff's α was computed across all three raters before any inference: **α = 0.84** (excellent, ≥ 0.80 threshold). Pairwise Pearson correlations: 0.87–0.92. This is the gate — if the raters don't agree, downstream effects are noise.

**Stage 3 — Bayesian hierarchical inference.** A PyMC partial-pooling model with publisher-level and passage-level random effects. Partial pooling regularizes small-sample publishers and lets us produce posterior distributions (not just point estimates) for each publisher's bias parameter. MCMC converged cleanly (**R-hat < 1.01, ESS > 3,000**).

**Stage 4 — Frequentist confirmation.** A non-parametric Friedman test as a sanity check on the rank-based effect across publishers: **χ² = 42.73, p < 0.001**.

#### Results that matter to a stakeholder

| Publisher | Posterior mean | 95% HDI | Verdict |
|-----------|---------------:|---------|---------|
| Publisher C | −0.48 | [−0.62, −0.34] | **Liberal lean — credible** |
| Publisher A | −0.29 | [−0.41, −0.17] | **Liberal lean — credible** |
| Publisher E | +0.02 | [−0.10, +0.14] | Neutral |
| Publisher B | +0.08 | [−0.04, +0.20] | Neutral |
| Publisher D | +0.38 | [+0.26, +0.50] | **Conservative lean — credible** |

- **3 of 5 publishers** had 95% HDIs that excluded zero — i.e., the model is statistically confident in the direction of the effect, not just the magnitude.
- **Cross-topic heatmap** localized the polarization: Social Issues showed the widest spread (Δ = 1.36 points between publishers), Mathematics showed essentially none — exactly the topical pattern a domain expert would predict.
- **Passage-level bootstrap CIs** flagged **12.3%** of passages as high-uncertainty and surfaced them for targeted human review — a hybrid LLM-plus-human workflow rather than full automation.

#### What I'd ship to a stakeholder
1. A **per-publisher report card** with the posterior mean, 95% HDI, and a "what would change my mind" decision rule.
2. A **topic-level drill-down** so a curriculum lead can see *where* the bias concentrates.
3. An **uncertainty triage queue** — the 12.3% high-uncertainty passages routed to human reviewers, with the LLM ensemble's full reasoning attached.
4. A **monitoring contract**: re-run quarterly, alert if any publisher's HDI shifts by > 0.15 from the prior estimate.

#### Data science skills demonstrated
- **Experimentation & inference:** Bayesian hierarchical modeling, partial pooling, posterior + HDI interpretation, multiple-testing-aware non-parametric tests, inter-rater reliability as a quality gate.
- **GenAI fluency:** LLM-as-judge ensembling, prompt engineering for rubric-based scoring, multi-model agreement as an uncertainty signal.
- **Engineering hygiene:** Circuit breakers, exponential backoff, MLflow experiment tracking, deterministic reruns with cached responses, audit trail.
- **Communication:** Findings framed as decisions (credible vs. neutral) rather than p-values; CI-aware recommendations for downstream reviewers.

#### Tech stack
`Python 3.12` `GPT-4o` `Claude-3.5-Sonnet` `Llama-3.2` `PyMC` `ArviZ` `SciPy` `statsmodels` `Pandas` `MLflow` `FastAPI` `LangChain`

---

### 🏥 Project 2 — Calibrated Binary Classification with Decision-Policy Tuning (WDBC)
**[📄 Technical Report](./Breast_Cancer_Classification_Report.md)** | **[📊 Publication PDF](./Breast_Cancer_Classification_Publication.pdf)**

> *In a high-stakes prediction problem with asymmetric error costs, the headline accuracy is the easy part. The real DS work is: which model, calibrated to what, with which threshold, for which downstream decision?*

#### The problem
The Wisconsin Diagnostic Breast Cancer (WDBC) dataset is a tabular binary classification problem with imbalanced classes and catastrophically asymmetric error costs — a false negative is a missed cancer, a false positive is an unnecessary biopsy. The clinical literature reports human inter-observer agreement of only **85–95%** on the same task. The DS question isn't "can we hit a high accuracy?" — it's "how do we choose, calibrate, and operate a model so that the *decision* it informs is defensible?"

#### The approach
**Step 1 — Preprocessing as a first-class step.**
- **VIF analysis** to identify and prune collinear features (radius, perimeter, area triplet collapsed to a single feature).
- **SMOTE** on the training fold only (never on holdout) to address the 63/37 class imbalance.
- **Recursive Feature Elimination (RFE)** with cross-validated scoring to pick the operating subset.

**Step 2 — Benchmark, don't single-model.** Eight ensemble algorithms compared under identical cross-validation: Random Forest, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM, Voting, Stacking. **AdaBoost won** — but the report shows the full leaderboard with overlap intervals, because a 0.2pp accuracy difference inside CV variance is not a real difference.

**Step 3 — Bayesian hyperparameter optimization.** Optuna's TPE sampler converged in **45 trials** vs. ~240 for grid search — 5× fewer fits for the same operating point. This is the practical, reproducible version of "Bayesian optimization" that DS teams should default to.

**Step 4 — Calibration, not just accuracy.** Raw classifier scores are not probabilities. **Platt scaling reduced Expected Calibration Error (ECE) from 0.0312 → 0.0089 — a 71.5% reduction.** This is the difference between a model whose "0.9 probability" actually means 90% and one whose scores are just rank-ordered.

**Step 5 — Decision-policy threshold tuning.** Rather than ship a single 0.5 cutoff, the report derives two operating points:
- **Mass screening:** threshold = 0.31 → **100% sensitivity** (zero missed cancers), specificity drops modestly.
- **Confirmation / second-read:** threshold = 0.62 → **100% precision** (zero false positives), recall stays at 98.59%.

The trade-off is explicit and tied to the cost ratio the stakeholder picks.

**Step 6 — Explainability.** SHAP values for global feature importance and per-prediction attributions. Concave-points-worst, perimeter-worst, and radius-worst dominated — consistent with cytopathology priors, which is the kind of sanity check a domain expert will ask for.

#### Headline results (held-out test, AdaBoost)

| Metric | Value | Why it matters |
|--------|------:|----------------|
| Accuracy | **99.12%** | Above the 85–95% human inter-observer band |
| Precision (PPV) | **100.00%** | Zero false positives at the chosen threshold |
| Recall (Sensitivity) | **98.59%** | One missed malignancy in 71 |
| Specificity | **100.00%** | Zero unnecessary biopsies at this threshold |
| F1 | **99.29%** | — |
| ROC-AUC | **0.9987** | Near-perfect discrimination |
| Cohen's κ | **0.9823** | Near-perfect agreement |
| Matthews CC | **0.9825** | Imbalance-robust scalar |
| 10-fold CV | **98.46% ± 1.12%** | Generalization is stable, not test-set luck |
| Calibration (ECE) | **0.0089** | Scores are actually probabilities |

#### What I'd ship to a stakeholder
1. **Two operating points** — screening vs. confirmation — with the cost ratio assumption written down so the choice is auditable.
2. A **calibrated probability output** (not just a label) so downstream workflows can route uncertain cases.
3. **Per-prediction SHAP attributions** so a reviewer sees *why* the model flagged a case.
4. A **drift monitor** on input feature distributions and on calibration ECE (not just accuracy), because calibration drifts faster than accuracy does.
5. A **model card** with the cohort, the metrics by subgroup, and the known failure modes — required for IEEE 2830-2025 / EU AI Act compliance in regulated domains.

#### Data science skills demonstrated
- **Modeling:** End-to-end tabular ML, ensemble benchmarking with CV-aware comparison, hyperparameter optimization (TPE), feature engineering with multicollinearity diagnostics.
- **Statistics:** Class-imbalance handling that respects the holdout, calibration discipline (ECE, reliability diagrams, Platt vs. isotonic), Cohen's κ / MCC for imbalance robustness.
- **Decision science:** Threshold tuning tied to asymmetric cost ratios, two-policy deployment, explicit decision rules instead of a black-box label.
- **Production:** MLflow registry, FastAPI serving (<100ms p95), drift monitoring on calibration not just accuracy, model card with subgroup audit.
- **Responsible AI:** SHAP attributions, fairness auditing, IEEE 2830-2025 alignment.

#### Tech stack
`Python 3.12` `scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `Optuna` `SMOTE (imbalanced-learn)` `SHAP` `MLflow` `FastAPI` `Docker`

---

### 📊 Cross-Project Synthesis

| | LLM Bias Detection | Calibrated Binary Classifier |
|---|---|---|
| **DS sub-discipline** | GenAI evaluation + Bayesian inference | Predictive modeling + decision policy |
| **Headline rigor signal** | Krippendorff's α = 0.84, R-hat < 1.01 | ECE = 0.0089, 10-fold CV stable |
| **Headline impact signal** | 3/5 publishers credibly biased (95% HDI) | 100% sensitivity at screening threshold |
| **Scale** | 67,500 ratings, 2.5M tokens, 4,500 passages | 569 samples, 30 features, 8 algos benchmarked |
| **What a reviewer can audit** | Posterior plots, HDI, prompt rubrics | Calibration plot, SHAP, threshold rationale |
| **Production artifact** | Quarterly publisher report card | Two-policy binary classifier with drift monitor |

Both projects share the same data-science backbone: **frame the decision first, choose statistics that match the decision, ship calibrated outputs and an audit trail, not just a metric**.

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
<td width="50%" valign="top">

### LLM Ensemble Bias Detection
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Inter-Rater Reliability** | α = 0.84 | Excellent (≥0.80) |
| **Pairwise Correlation (GPT-4o ↔ Claude)** | r = 0.92 | Near-perfect |
| **Model Convergence (R-hat)** | < 1.01 | Perfect |
| **Effective Sample Size** | > 3,000 | Adequate posterior |
| **Statistical Power** | χ² = 42.73 | p < 0.001 |
| **Credible Findings** | 3 of 5 publishers | 95% HDI excludes 0 |
| **Scale** | 67.5K ratings / 2.5M tokens | Production |
| **Effect Size Range** | −0.48 to +0.38 | On [−2, +2] scale |

</td>
<td width="50%" valign="top">

### Calibrated Binary Classifier (WDBC)
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 99.12% | Above 85–95% human band |
| **Precision (PPV)** | 100.00% | Zero false positives |
| **Recall (Sensitivity)** | 98.59% | 1 miss in 71 |
| **Specificity** | 100.00% | Zero unnecessary biopsies |
| **ROC-AUC** | 0.9987 | Near-perfect discrimination |
| **Cohen's κ / Matthews CC** | 0.9823 / 0.9825 | Imbalance-robust |
| **10-fold CV** | 98.46% ± 1.12% | Stable generalization |
| **Calibration (ECE)** | 0.0089 | After Platt scaling (71.5% ↓) |
| **Optuna trials to converge** | 45 (vs 240 grid) | 5× fewer fits |

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
✅ **Communication**: 2 publication-grade reports with model cards, calibration plots, posterior visualizations, and SHAP attributions written for non-technical readers  
✅ **Domain Versatility**: Healthcare (high-stakes decision modeling) and education (GenAI evaluation at scale) — comfortable in regulated environments

---

## 📚 Publications & Technical Reports

| Title | Type | Date | Links |
|-------|------|------|-------|
| **LLM Ensemble Textbook Bias Detection** | Technical Report v4.0.0 | Apr 2026 | [Report](./LLM_Ensemble_Bias_Detection_Report.md) • [PDF](./LLM_Bias_Detection_Publication.pdf) |
| **Calibrated Binary Classification (WDBC)** | Technical Report v4.0.0 | Apr 2026 | [Report](./Breast_Cancer_Classification_Report.md) • [PDF](./Breast_Cancer_Classification_Publication.pdf) |

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
Data-Science-Portfolio/
├── 📄 README.md                                           # This portfolio
├── 📄 Resume_Derek_Lankeaux.md                            # Resume
├── 🔬 LLM_Ensemble_Bias_Detection_Report.md               # Project 1: GenAI eval + Bayesian inference
├── 📑 LLM_Bias_Detection_Publication.pdf                  # Project 1 publication PDF
├── 🏥 Breast_Cancer_Classification_Report.md              # Project 2: calibrated binary classifier
├── 📑 Breast_Cancer_Classification_Publication.pdf        # Project 2 publication PDF
├── 🐍 generate_publication_pdfs.py                        # PDF build script
└── 📁 latex/                                              # LaTeX sources
```

---

### 🔍 Keywords for Search & ATS

</div>

**Data Science:** Experimentation • A/B Testing • Causal Inference • Quasi-Experiments • Metric Design • Cohort Analysis • Funnel Analysis • Power Analysis • Hypothesis Testing • Stakeholder Communication • SQL • Dashboarding

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Bagging • Feature Engineering • Model Calibration • Threshold Tuning

**Large Language Models:** GPT-4 • GPT-4o • Claude-3.5-Sonnet • Llama-3.2 • BERT • Transformers • Prompt Engineering • Few-Shot Learning • Zero-Shot Learning • In-Context Learning • LLM-as-Judge • RAG Evaluation

**Bayesian Statistics:** Hierarchical Modeling • MCMC • PyMC • Stan • Posterior Inference • Prior Specification • Credible Intervals • Bayesian Inference • Probabilistic Programming • HDI

**Statistical Methods:** Hypothesis Testing • Cross-Validation • Bootstrap • Permutation Testing • Effect Sizes • Power Analysis • Multiple Testing Correction • Inter-Rater Reliability • Krippendorff's Alpha • Cohen's Kappa

**Explainable AI (XAI):** SHAP • LIME • Feature Importance • Model Interpretability • Fairness Auditing • Bias Detection • Responsible AI • AI Ethics • AI Governance • Audit Trails

**MLOps & Production:** MLflow • Weights & Biases • Model Registry • Experiment Tracking • FastAPI • Docker • Kubernetes • CI/CD • Model Monitoring • Drift Detection • A/B Testing • Circuit Breakers

**Programming:** Python • R • SQL • PyTorch • TensorFlow • scikit-learn • Pandas • NumPy • Dask • Apache Spark

**Research Engineering:** Technical Writing • Statistical Validation • Reproducible Research • Peer Review • Literature Review • Experimental Design • Causal Inference • Cost-Benefit Analysis

**Standards & Compliance:** IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act • GDPR • Model Cards • Transparency • Accountability • AI Governance

---

<div align="center">

**📌 Last Updated:** April 2026  
**✅ Compliance:** IEEE 2830-2025 (Transparent ML) • ISO/IEC 23894:2025 (AI Risk Management)  
**🔒 License:** Portfolio content © 2026 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
