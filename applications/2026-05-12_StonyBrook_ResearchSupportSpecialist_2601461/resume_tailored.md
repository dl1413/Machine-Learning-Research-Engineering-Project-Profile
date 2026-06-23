# Derek Lankeaux, MS (in progress)

**Research Support Specialist Candidate | Applied Statistics | Python & R | Reproducible Data Pipelines**

[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Summary

Applied-statistics researcher with a track record of harmonizing large,
heterogeneous datasets, building reproducible analysis pipelines in
Python and R, and writing publication-quality technical reports.
Strongest in Bayesian hierarchical modeling (PyMC, MCMC), data
management and visualization at scale, and end-to-end Linux / Jupyter /
GitHub workflows. Three independent research projects published as
technical reports in the last quarter.

---

## Education

**Master of Science in Applied Statistics** — Rochester Institute of Technology — Expected 2026
*Specialization: Bayesian Methods, Statistical Learning, Experimental Design*
Relevant coursework: Advanced Bayesian Inference & MCMC, Statistical Learning Theory, Experimental Design & Causal Inference, High-Dimensional Statistics, Computational Statistics & Optimization

---

## Technical Skills (mapped to the position requirements)

- **Programming:** Python 3.12+ (pandas, polars, numpy, scikit-learn, PyMC, ArviZ), R, SQL, Bash
- **Statistics & data analysis:** Bayesian hierarchical modeling, MCMC diagnostics (R-hat, ESS), partial pooling, posterior credible intervals (95% HDI), Friedman / Nemenyi tests, multicollinearity diagnostics (VIF), hypothesis testing
- **Data management & visualization (large datasets):** harmonization across heterogeneous data streams, MLflow lineage, schema-driven preprocessing, ArviZ posterior plots, matplotlib / seaborn
- **Reproducible-science toolchain:** Jupyter Notebooks, GitHub, Linux (CLI, shell scripting), HPC-compatible Python environments, Docker
- **Cloud / collaboration:** AWS, GCP, MLflow model registry; comfortable adding Box to the workflow
- **Currently ramping (transferable foundations in place):** geopandas, rasterio, Google Earth Engine, QGIS — applying the same reproducible-pipeline habits to spatial data

---

## Research Experience

### Independent Research — Multi-Source Bayesian Hierarchical Analysis (January 2026)
*Closest analog to the cropland / livestock harmonization workflow described in the JD.*

- Harmonized **67,500 rating records over 4,500 source documents (2.5M tokens)** drawn from heterogeneous sources into a single analysis-ready dataset under a documented schema
- Fit **PyMC Bayesian hierarchical model with partial pooling** across publisher groups; achieved MCMC convergence with R-hat < 1.01 and ESS > 1000
- Produced **95% HDI credible intervals** per group and ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) with post-hoc Nemenyi pairwise comparisons to localize effects
- Engineered async API integration with circuit breakers and exponential backoff sustaining 67,500 records without manual intervention
- Delivered a **research-quality technical report** covering literature review, methods, priors, sensitivity analysis, and posterior visualizations

**Tech stack:** Python, PyMC, ArviZ, pandas, MLflow, Jupyter, GitHub, Linux

---

### Independent Research — Multi-Stream Data Harmonization & Evaluation Framework (January 2026)

- Built **production data pipeline harmonizing outputs from three independent rater streams** over 12,500 paired records and 6 categorical labels into a unified evaluation schema
- Engineered **47 features** spanning linguistic, semantic, and structural signals — a parallel to the kind of feature derivation needed when reconciling cropland, livestock, and food-supply layers
- Achieved 96.8% classification accuracy and ROC-AUC 0.9923; documented inter-rater reliability at Krippendorff's alpha = 0.81
- Implemented Bayesian hierarchical analysis to quantify per-source uncertainty with 95% HDI intervals
- Built MLOps infrastructure: MLflow experiment tracking, SHAP attribution, full audit trail

**Tech stack:** Python, PyMC, XGBoost, scikit-learn, MLflow, Jupyter, GitHub, Linux

---

### Independent Research — Reproducible Classification Pipeline with Statistical Rigor (January 2026)

- Benchmarked **8 algorithms under nested cross-validation** on a high-stakes binary classification task; documented selection criteria end-to-end
- Applied **VIF-based multicollinearity diagnostics, SMOTE class balancing, and RFE feature selection** — the same hygiene applied to multicollinear environmental covariates
- Achieved 99.12% accuracy with 100% precision, 98.59% recall, ROC-AUC 0.9987 on the held-out set
- Productionized winning model with FastAPI and MLflow model registry; <100ms p95 latency

**Tech stack:** Python, scikit-learn, XGBoost, LightGBM, SHAP, MLflow, Linux

---

## Publications & Technical Reports

| Title | Type | Date |
|-------|------|------|
| Multi-Source Bayesian Hierarchical Analysis (LLM Ensemble Bias Detection) | Technical Report v3.0.0 | January 2026 |
| Multi-Stream Evaluation Framework (AI Safety Red-Team Evaluation) | Technical Report v1.0.0 | January 2026 |
| Reproducible Classification Pipeline (Breast Cancer Classification) | Technical Report v3.0.0 | January 2026 |

All three reports include literature reviews, transparent methodology
sections, statistical sensitivity analyses, and reproducible code — the
same standard expected for the project reports and peer-reviewed
manuscripts described in the position.

---

## Core Competencies (aligned to the position)

- **Statistics & geospatial-adjacent analytics:** Bayesian hierarchical modeling, MCMC diagnostics, partial pooling, credible intervals, hypothesis testing — directly applicable to multi-region cropland and livestock-density modeling
- **Data harmonization & large-dataset management:** schema-driven preprocessing, lineage tracking, reproducible pipelines, async data ingestion at scale
- **Reproducible-science toolchain:** Python + R, Jupyter, GitHub, Linux / HPC-ready environments, MLflow
- **Technical writing:** three publication-quality reports in one quarter; comfortable with literature reviews, methods sections, and manuscript drafting under PI review

---

**Location:** Open to on-site at Stony Brook, NY; also open to hybrid
**Timeline:** Available for a 2026 start once MS in Applied Statistics is conferred
**Work Authorization:** Authorized to work in the United States
