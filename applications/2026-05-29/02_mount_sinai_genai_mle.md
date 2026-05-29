# 2. Mount Sinai — Machine Learning Engineer III, Generative AI

Windreich Department of Artificial Intelligence & Human Health Research

- **Location:** New York, NY
- **Submit:** https://careers.mountsinai.org/jobs/3035197
- **Role family:** Healthcare / Clinical ML + GenAI
- **Lead project:** P3 — Clinical-Grade Breast Cancer Classification · **Supporting:** P1 (GenAI eval rigor), P2

## JD keywords to echo (verbatim)
generative AI · LLM · clinical · healthcare · model deployment · medical data · Python · cross-functional · explainability

## Cover letter

Dear Windreich Department Hiring Team,

The work most relevant to this role is a **clinical**-grade breast-cancer
classifier I shipped this year. I benchmarked 8 algorithms end-to-end — Random
Forest through stacking ensembles — and landed at **99.12% accuracy** with 100%
precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987, comfortably
above the 90–95% range typically cited for human expert reads. Just as important
for **healthcare** deployment: the pipeline ships with SHAP **explainability**
per prediction, VIF-pruned features, and **model deployment** behind a FastAPI
service under 100ms p95, aligned with IEEE 2830-2025 transparency standards.

Because the Windreich Department pairs clinical ML with **generative AI**, my
second project maps directly: an independent **LLM** red-team evaluation
framework (GPT-4o, Claude-3.5, Llama-3.2) that auto-grades 12,500 response pairs
at 96.8% accuracy with audit-grade reliability (Krippendorff's alpha = 0.81). It
shows I can bring the same statistical rigor I apply to **medical data** to
evaluating generative systems before they touch patient-facing workflows — and
I'm comfortable owning the full **Python** stack and collaborating
**cross-functional**ly with clinicians and researchers.

I'm based in / available for New York City, targeting a 2026 start once I wrap my
MS in Applied Statistics at RIT. Code and three technical reports are on my
GitHub (dl1413).

Best,
Derek Lankeaux

## Resume — Projects section (lead order for this role)

**Clinical-Grade Breast Cancer ML Classification** — *Independent Research, Jan 2026*
- Built clinical-grade classifier at 99.12% accuracy / 100% precision (zero false positives) / 98.59% recall / ROC-AUC 0.9987, exceeding the 90–95% human-expert range
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) under nested cross-validation; VIF multicollinearity pruning, SMOTE balancing, RFE selection
- Implemented per-prediction SHAP explanations for clinical transparency (IEEE 2830-2025); deployed via containerized FastAPI + MLflow registry at < 100ms p95

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) grading 12,500 pairs at 96.8% accuracy, ROC-AUC 0.9923; Krippendorff's alpha = 0.81; PyMC 95% HDI uncertainty

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- 67,500 ratings / 2.5M tokens; Bayesian hierarchical model (R-hat < 1.01); significant bias at p < 0.001
