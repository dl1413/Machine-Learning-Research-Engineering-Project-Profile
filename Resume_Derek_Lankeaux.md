# Derek Lankeaux, MS

**Data Scientist | Applied Statistician | LLM Evaluation & Bayesian Inference**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Applied-statistics MS shipping end-to-end data science: framing the estimand, designing the experiment, writing the SQL, fitting the model, quantifying uncertainty, and writing the readout for non-technical stakeholders. Three published technical reports cover multi-LLM evaluation (Krippendorff's alpha 0.81-0.84), Bayesian hierarchical inference (R-hat < 1.01, p < 0.001), and clinical decision support (99.12% accuracy, 100% precision on WDBC). All three are reproducible, MLflow-tracked, and aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act.

---

## Technical Skills

**Languages:** Python 3.12+, R, SQL, Bash

**Experimentation & Statistics:** A/B testing, power analysis, hypothesis testing, multiple-testing correction (Bonferroni, FDR), effect sizes (Cohen's d, eta-squared), bootstrap CIs, quasi-experimental design, inter-rater reliability (Krippendorff's alpha, Cohen's kappa, Friedman)

**Bayesian Inference:** PyMC 5.15+, ArviZ 0.18+, NumPyro; hierarchical models with partial pooling, MCMC diagnostics (R-hat, ESS, MCSE), 95% HDI, posterior predictive checks, sensitivity analysis

**ML & Modeling:** scikit-learn 1.5+, XGBoost 2.1+, LightGBM 4.5+, AdaBoost, Stacking/Voting; VIF, SMOTE (train-only), RFE, threshold tuning, cost-sensitive learning, calibration (Platt, isotonic, Brier)

**GenAI / LLM Evaluation:** GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B; LLM-as-judge ensembles, rubric design, structured-JSON output, prompt-versioned offline benchmarking

**Data Stack:** SQL, Pandas 2.2+, Polars 1.0+, NumPy 2.0+, Apache Arrow

**MLOps & Deployment:** MLflow 2.15+, FastAPI 0.110+, Docker, AWS, GCP, DVC

**Explainability & Responsible AI:** SHAP (TreeExplainer), LIME, Fairlearn, model cards; IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act Article 13

---

## Education

**Master of Science in Applied Statistics**
Rochester Institute of Technology | Expected 2026
*Concentration: Bayesian Methods, Statistical Learning, Experimental Design*

**Relevant Coursework:** Advanced Bayesian Inference & MCMC, Statistical Learning Theory, Experimental Design & Causal Inference, High-Dimensional Statistics, Computational Statistics & Optimization, Deep Learning

---

## Research Projects

### AI Safety Red-Team Evaluation Framework — *Report AI-SR-2026-01*
*Independent Research Project | January 2026*

- Built a dual-stage evaluation pipeline: three frontier LLMs (GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B) annotate 12,500 prompt-response pairs across 6 harm categories; a stacking meta-learner scales those judgments at audit-grade reliability (Krippendorff's alpha = 0.81)
- Stacking classifier reaches 96.8% accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923, and FNR 3.9% on a held-out fold (n = 2,500); wins on identical 10-fold splits against 7 alternative ensembles
- Engineered 47 features (lexical, semantic, structural, safety-specific, embedding-derived) with VIF pruning, train-fold-only SMOTE, and RFE -- no SMOTE leakage into the test fold
- Fit a PyMC hierarchical risk layer (model-class x category, with interaction) producing 95% HDI per population estimate; R-hat < 1.01, ESS > 3,000
- Pipeline runs at ~850 samples/hr at $0.018/sample -- a 340x marginal cost reduction vs. human review ($6.12/sample) -- with circuit breakers, exponential backoff, MLflow lineage, SHAP attribution per prediction, and a FastAPI scoring stub at p95 < 100 ms

**Stack:** Python, scikit-learn, XGBoost, LightGBM, PyMC, ArviZ, SHAP, MLflow, FastAPI

---

### LLM-Ensemble Textbook Bias Detection — *Report AI-SR-2026-02*
*Independent Research Project | January 2026*

- Sharpened a vague public-debate question into a measurable estimand: posterior contrast of per-publisher ideological lean against neutral, after accounting for per-rater calibration and finite-sample noise
- Collected 67,500 ratings from a three-LLM ensemble (GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B) over 4,500 passages drawn from 150 textbooks across 5 U.S. publishers (~2.5M tokens) at temperature 0.3 with a 5-dimension rubric
- Validated the instrument first: Krippendorff's alpha = 0.84; pairwise correlations 0.87-0.92; rater x publisher interaction not credible under partial pooling
- Fit a PyMC hierarchical model (passages within subjects within publishers, with rater-specific intercepts); 4 chains x 4,000 post-warmup draws; R-hat < 1.01, ESS > 3,000, MCSE < 0.005
- Reported posterior means and 95% HDIs per publisher; 3 of 5 publishers show credible non-neutral lean (two liberal, one conservative). Friedman chi-squared = 42.73, p < 0.001 confirms; rank order stable across binary, 5-point, and [-2, +2] scale operationalizations

**Stack:** Python, PyMC, ArviZ, LangChain, Polars, MLflow, tenacity

---

### Ensemble Classifier for the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset — *Report AI-SR-2026-03*
*Independent Research Project | January 2026*

- Reframed screening as a utility-maximization decision: published a cost-ratio threshold sweep (miss / FP in [1, 100]) rather than relying on a default 0.5 cutoff
- Benchmarked 8 ensembles -- Random Forest, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM, Voting, Stacking -- on identical stratified 80/20 folds (n = 569, 30 features) with a shared preprocessing pipeline: z-score scaling fit on train only, VIF review, train-fold-only SMOTE (k = 5), and RFE to 15 features
- Winner -- AdaBoost -- reaches 99.12% accuracy, 100% precision (zero false positives on the held-out fold), 98.59% recall, F1 = 99.29%, ROC-AUC = 0.9987, Cohen's kappa = 0.9823, MCC = 0.9825
- 10-fold stratified CV confirms 98.46% +/- 1.12%; 95% CI [96.27%, 100.65%]; coefficient of variation 1.14%
- SHAP (TreeExplainer) attribution coheres with cytopathological intuition -- worst-region extreme-value features (worst concave points, worst perimeter, worst radius) dominate over mean-region features
- Limitations explicit and in-section: WDBC is a benchmark dataset, n = 569 is small for fairness slicing, no prospective validation -- deployment claim is out of scope and stated as such

**Stack:** Python, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, MLflow, FastAPI, Fairlearn

---

## Publications & Technical Reports

| Report | Title | Date |
|---|---|---|
| AI-SR-2026-01 | Multi-LLM Ensemble Annotation and Bayesian Classification for AI Safety Red-Team Evaluation | January 2026 |
| AI-SR-2026-02 | Causal Bias Analysis in K-12 Textbooks: A Multi-LLM Hierarchical Study | January 2026 |
| AI-SR-2026-03 | Enhanced Ensemble Methods for Wisconsin Breast Cancer Classification | January 2026 |

---

**Location:** Available for remote/hybrid positions
**Timeline:** Seeking 2026 start
**Work Authorization:** Authorized to work in the United States
