# Cover Letter — Memorial Sloan Kettering, ML Scientist / Engineer (Computational Oncology)

Dear MSK Computational Oncology Hiring Team,

I'm applying for a Machine Learning Scientist / Engineer role in Computational
Oncology. The expanded campus under Dr. Shah's leadership — Computational
Oncology, Radiology Informatics, Clinical Bioinformatics, HPC — is the kind of
place where the work I want to do (rigorous clinical ML with calibrated
uncertainty) is the central work, not the side project.

The most relevant evidence is a clinical-grade breast-cancer classifier I
shipped this year. I benchmarked 8 algorithms end-to-end — Random Forest,
XGBoost, LightGBM, AdaBoost, Stacking, Voting — under nested cross-validation,
and landed at **99.12% accuracy with 100% precision (zero false positives),
98.59% recall, and ROC-AUC 0.9987**, comfortably above the 90-95% range
typically cited for human expert reads. Just as important for clinical
deployment: the pipeline ships with SHAP per-prediction explanations, VIF-pruned
features (multicollinearity diagnostics), SMOTE class-balancing, and a FastAPI
service under 100ms p95, all aligned with IEEE 2830-2025 transparency
standards. The codebase, tests, and report are reproducible from a single
clone.

Two other projects show the breadth I'd bring beyond a single classifier. A
3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) for harm classification that
hit 96.8% accuracy at 340x lower cost than human annotation — same auditability
discipline (Krippendorff's alpha = 0.81, MLflow lineage end-to-end), and the
beginnings of a template for LLM-assisted chart review or radiology-report
triage. And a Bayesian hierarchical bias study (67,500 ratings, 4,500 passages)
with R-hat < 1.01 and 95% HDI credible intervals — the partial-pooling habit
maps cleanly to multi-site / multi-cohort clinical data, where you want to
share statistical strength without erasing site-level heterogeneity.

What I'd want to bring to Computational Oncology: a Bayesian-first reflex
when point estimates won't survive a tumor board, an instinct for shipping
explanations alongside predictions, and the engineering discipline to put
both behind a service that clinicians can actually use. NYC-based, 2026 start
once I wrap my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux
