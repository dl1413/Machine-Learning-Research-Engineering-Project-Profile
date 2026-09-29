# Clinical Privacy vs. Predictive Utility

**Type:** Independent technical case study
**Focus:** Re-identification risk and model utility on real hospital discharge data
**Data:** Diabetes 130-US Hospitals, 1999–2008 (UCI ML Repository, ID 296)

| Read | Link |
|---|---|
| Portfolio overview | [README](../../README.md) |
| Pipeline | Built as a 13-section pipeline on Zerve; notebook available on request |

## Project at a glance

| Problem | Approach | Reported evidence |
|---|---|---|
| HIPAA Safe Harbor removes explicit identifiers, but quasi-identifiers left in the record (demographics, payer, specialty, admission codes) can still re-identify patients. Which mitigation gives defensible privacy without breaking the model that has to run on the data? | Measure re-identification risk with k-anonymity, l-diversity, t-closeness, and a differential-privacy ε-utility curve; predict 30-day readmission under each privacy regime with logistic regression and tuned histogram gradient boosting; audit subgroup performance and decision thresholds. | 101,766 encounters from 71,518 patients at 130 US hospitals; overall ROC-AUC 0.672; unique-record risk cut from 15.4% to 7.6% by generalization. |

## Results

| Strategy | Records | Unique records (k = 1) | k < 5 | Gradient boosting AUC |
|---|---|---|---|---|
| Baseline (raw) | 101,766 | 15.35% | 31.10% | 0.673 |
| **Generalized** | 101,766 | **7.56%** | 17.63% | 0.670 |
| Suppressed (k ≥ 5) | 70,112 (−31.1%) | 0.00% | 0.00% | 0.651 |

- **Generalization is the right default.** Coarsening age into three bands and collapsing rare specialty and payer codes halves unique-record risk at a 0.3-point AUC cost, with no records dropped.
- **Suppression is the publication tier.** It removes unique records by definition but discards 31% of patients and costs 2.2 AUC points.
- **The model fails for the oldest patients.** Overall AUC is 0.672, but for patients aged 90–100 it falls to 0.515, barely above chance. Performance across race groups is nearly identical (0.001 AUC spread).
- **The default threshold is unusable.** At 0.50 the model's recall is 0.2%; the F1-optimal threshold is 0.126.
- **k-anonymity is not enough.** After suppression, 59% of groups still exceed t = 0.15 for t-closeness. Post-hoc Laplace noise destroys utility at ε ≤ 1, so a formal privacy guarantee at useful utility needs training-time DP-SGD.
- **Two features carry most of the signal.** Permutation importance ranks prior inpatient visits and discharge disposition far above everything else.

## What this demonstrates

- Evaluation that goes past the aggregate metric: subgroup AUC, threshold analysis, and permutation importance.
- Patient-level train/test splits (GroupShuffleSplit) to prevent leakage across encounters from the same patient.
- A concrete recommendation with its trade-off stated: generalize for internal analytics, suppress for external release.

## Scope

This is a retrospective study on a public research dataset. The readmission
model is not validated for clinical use, and the age-band failure would block
deployment for patients aged 90 and over without further work.

## Methods

`k-anonymity` · `l-diversity` · `t-closeness` · `differential privacy` · `scikit-learn` · `HistGradientBoosting` · `permutation importance` · `Zerve`
