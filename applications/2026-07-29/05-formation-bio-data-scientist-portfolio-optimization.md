# Formation Bio — Data Scientist, Portfolio Optimization

**Location:** New York, NY (hybrid, 3 days/week in office)
**Apply URL:** https://job-boards.greenhouse.io/formationbio/jobs/7757667
**Team context:** Formation Bio is an NYC-headquartered AI-first drug-development company; the Portfolio Optimization DS role sits at the intersection of clinical evidence, decision science, and program prioritization.

---

## Cover Letter

Dear Formation Bio Hiring Team,

I'm writing about the Data Scientist, Portfolio Optimization role. Portfolio decisions in drug development live and die by the quality of the evidence underneath them — calibrated probabilities, honest uncertainty, and a model whose recommendations survive review. That's the shape of the work I've been doing, and it's the reason I'd like to bring it to Formation Bio.

The most directly relevant project is a **clinical-grade Breast Cancer classification system**. Across an **8-algorithm benchmark** (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting), the best model reached **99.12% accuracy with 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987** — and, more importantly for a portfolio-decision context, **Platt scaling brought the ECE from 0.0312 to 0.0089 (71.5% reduction)** so downstream thresholds could be tuned to a decision policy (**100% sensitivity at 0.31 for screening**) instead of a leaderboard score. **Optuna TPE converged in 45 trials vs. 240 for grid search**, SHAP carried the interpretability layer, and MLflow held the versioned artifacts — the kind of reproducibility a drug-portfolio decision has to inherit.

The Bayesian side of the toolkit shows up in my **LLM-Ensemble Bias Detection** work — a **PyMC hierarchical partial-pooling model (R-hat < 1.01, 95% HDI)** on **67,500 ratings across 4,500 units**, which is essentially the same posterior-decision problem as ranking programs by expected value under uncertainty. Multiple-testing correction, bootstrap CIs for passage-level uncertainty, and inter-rater reliability (**Krippendorff's α = 0.84**) are all in that project's pipeline.

I've also worked at production LLM-eval scale (**AI Safety Red-Team: 12,500 samples, 6 categories, 96.8% accuracy, $0.018/sample, α = 0.81**) — useful if the team is folding LLM-based summarization or literature extraction into portfolio scoring. I'm finishing my MS in Applied Statistics at RIT and would welcome a conversation about the team's current portfolio-optimization workstream.

Regards,
Derek Lankeaux
LinkedIn: https://linkedin.com/in/derek-lankeaux · GitHub: https://github.com/dl1413

---

## Role-tailored resume bullets

- Clinical-grade ensemble at **99.12% accuracy / 100% precision / ROC-AUC 0.9987**, with **Platt scaling cutting ECE by 71.5% (0.0312 → 0.0089)** and threshold policies tuned to context — calibration a portfolio decision can inherit.
- Posterior-based ranking on **67.5K noisy ratings** using a **PyMC hierarchical partial-pooling model (R-hat < 1.01, 95% HDI)** — the same shape as ranking programs by expected value under uncertainty.
