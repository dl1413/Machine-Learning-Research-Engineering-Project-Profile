# 05 — Flatiron Health · Senior ML Engineer, Real-World Evidence

| Field | Value |
|---|---|
| **Company** | Flatiron Health |
| **Role** | Senior Machine Learning Engineer — Real-World Evidence (or "Oncology Data Science") |
| **Location** | NYC HQ / US-remote |
| **Source** | flatiron.com/careers (apply direct) |
| **JD link** | _VERIFY before submit:_ https://flatiron.com/careers/ (filter: ML / Data Science / RWE) |
| **Lead project** | P3 — Breast Cancer Classification (clinical-grade tabular ML) |
| **Supporting** | P2 — LLM Bias (Bayesian inference on real-world labels); P1 — Red-Team (eval + audit) |
| **Resume version** | `resume_v3_healthcare.pdf` |
| **Cover letter** | Yes — see below |
| **Referral** | Search LinkedIn for Flatiron oncology DS leads — warm intro if 2nd-degree |

## JD keyword echo (fill in after reading JD)

- [ ] real-world evidence / EHR / claims
- [ ] oncology / tumor / outcomes
- [ ] feature engineering / model calibration
- [ ] [Flatiron-specific: "abstraction", "OncoEMR", "RWE study"]
- [ ] [Flatiron-specific phrase 2]

## Resume bullets (top 4 — paste into "Selected Projects")

From `APPLICATION_SNIPPETS.md` → P3 → "Applied ML / MLE" + "Healthcare / Clinical ML":

- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) and shipped winning ensemble at 99.12% accuracy / ROC-AUC 0.9987
- Applied VIF-based multicollinearity pruning, SMOTE class-balancing, and RFE feature selection to lift recall to 98.59% while holding precision at 100%
- Deployed as a containerized FastAPI service with MLflow model registry; p95 latency under 100ms; SHAP explanations per prediction (IEEE 2830-2025)
- Built parallel LLM-ensemble study at 67,500 ratings with Bayesian hierarchical uncertainty (R-hat < 1.01) — transferable to label-noise / abstraction-noise modeling on EHR-derived datasets

## Cover letter draft

Dear Flatiron Health team,

A recent project that captures how I work: I built a clinical-grade
ensemble classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987)
by benchmarking 8 algorithms — Random Forest, XGBoost, LightGBM,
AdaBoost, Stacking, Voting — under nested cross-validation, then
productionized the winner behind a FastAPI service at <100ms p95.
Calibration was treated as a feature: Platt scaling brought ECE to
0.0089, with context-adaptive thresholds (screening vs. confirmatory)
backed by SHAP explanations for clinical transparency. That habit —
pick by evidence not fashion, calibrate and explain by default —
is what I'd bring to Flatiron's RWE modeling.

The other two projects round out the profile for oncology data work
specifically. An LLM-ensemble bias study (67,500 ratings, 4,500
passages, Bayesian hierarchical with R-hat < 1.01) shows I can model
label noise and rater disagreement rigorously — the same statistical
shape as inter-abstractor variability in EHR data. And an
audit-grade AI safety eval framework (Krippendorff's α = 0.81,
SHAP, IEEE 2830-2025 provenance) shows I take governance seriously,
which I expect matters for any model that touches regulated
oncology outcomes.

I'm finishing an MS in Applied Statistics at RIT and based in /
available for NYC, also open to remote. Portfolio, code, and the
three technical reports are on my GitHub (dl1413). Happy to walk
through any of them.

Best,
Derek Lankeaux
