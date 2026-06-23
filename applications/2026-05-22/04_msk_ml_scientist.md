# Memorial Sloan Kettering — Machine Learning Scientist

- **Location:** New York City (on-site/hybrid)
- **Source:** careers.mskcc.org (req 2025-86362)
- **Lead project:** P3 — Clinical-Grade Breast Cancer ML Classification
- **Supporting:** P1 (statistical rigor / uncertainty), P2 (Bayesian)
- **Fit note:** Strong on clinical rigor; MSK leans medical imaging / computational
  pathology + HPC. Lead with clinical-grade results and statistical depth; be honest
  that imaging-specific deep learning is adjacent, not core, experience.
- **JD phrases to echo:** computational pathology, deep learning, clinical integration,
  statistical methods, production-ready AI tools, partner with clinicians.

## Resume — Projects section order
1. Clinical-Grade Breast Cancer ML Classification (lead)
2. AI Safety Red-Team Evaluation (uncertainty quantification)
3. LLM Ensemble Textbook Bias Detection (Bayesian)

### Top resume bullets
- Developed clinical-grade ensemble (99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987) exceeding the 90–95% human-expert range on breast-cancer classification
- Applied VIF multicollinearity diagnostics, SMOTE balancing, RFE selection, and stratified CV; **statistical methods** chosen by evidence, not fashion
- Built **production-ready AI tools**: containerized FastAPI service, MLflow registry, <100ms p95, SHAP per-prediction explanations for **clinical integration**
- Quantified multi-model uncertainty via PyMC Bayesian hierarchy (95% HDI) — defensible risk intervals for high-stakes decisions

### Skills to surface
scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, Bayesian/PyMC, FastAPI, MLflow, Docker/Kubernetes, HPC-friendly pipelines.

## Cover letter
> Dear MSK Computational Oncology team,
>
> The work most relevant to your Machine Learning Scientist role is a clinical-grade
> breast-cancer classifier I built and published this year. Benchmarking 8 algorithms
> under stratified cross-validation, I landed at 99.12% accuracy with 100% precision
> (zero false positives), 98.59% recall, and ROC-AUC 0.9987 — comfortably above the
> 90–95% range cited for expert reads — and shipped it as a **production-ready AI
> tool**: a FastAPI service under 100ms p95 with an MLflow registry and SHAP
> explanations per prediction to support **clinical integration** and transparency.
>
> Underneath the headline numbers is the part I care most about and that I think MSK
> values: rigorous **statistical methods**. VIF multicollinearity pruning, SMOTE,
> RFE, and a PyMC Bayesian hierarchical layer producing 95% HDI intervals so a
> prediction comes with quantified uncertainty. I'd be glad to extend that rigor to
> computational-pathology and imaging problems alongside your clinicians; I'd note
> imaging-specific deep learning is adjacent to my core work rather than my deepest
> area, and I'm a fast study there.
>
> I'm completing an MS in Applied Statistics at RIT (2026), NYC-based, US
> work-authorized. Portfolio and three technical reports on GitHub (dl1413). Salary
> open, targeting market for the role/location.
>
> Best, Derek Lankeaux · dlankeaux12@gmail.com · linkedin.com/in/derek-lankeaux
