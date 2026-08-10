# Cover Letter Template — Data Scientist / Quantitative Analyst / Research Scientist (non-LLM)

**Best fit for:** NYC quant/finance (Two Sigma, Jane Street research roles, Bloomberg quant, Citadel), healthcare analytics (Memorial Sloan Kettering data science, NYC Health + Hospitals, Tempus, Flatiron Health), causal-inference / experimentation teams (Spotify, Netflix, Etsy).

**Anchor:** Lead with Bayesian statistics depth + Breast Cancer (classical ML rigor). LLM work as a secondary signal of technical range.

---

Dear {{Hiring Manager Name | Hiring Team}},

I'm applying for the **{{Role Title}}** position at **{{Company}}**. {{Specific reference — a paper from the team, a methodology they're known for, a product analytics blog post}}.

I'm an MS Applied Statistics candidate at RIT (2026) specializing in Bayesian methods and experimental design. My recent independent research has hit both ends of the rigor spectrum — clinical-grade classical ML and modern LLM evaluation — using the same statistical toolkit:

- **Clinical-Grade Breast Cancer Classifier** — 99.12% accuracy, 100% precision, ROC-AUC 0.9987 across an 8-algorithm benchmark (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting). Pipeline: VIF multicollinearity analysis → SMOTE balancing → RFE feature selection → Optuna TPE hyperparameter search (45 vs. 240 trials) → Platt calibration (ECE 0.0312 → 0.0089). Cross-validation stability 98.46% ± 1.12%.

- **LLM Ensemble Bias Detection** — PyMC hierarchical Bayesian model with partial pooling across 5 publishers and 4,500 passages. MCMC convergence verified (R-hat < 1.01, effective sample size analyses). Identified credible publisher-level bias with Friedman χ² = 42.73, p < 0.001, and 95% HDI uncertainty intervals at the passage level (12.3% flagged as high-uncertainty for expert review).

- **AI Safety Red-Team Evaluation** — Inter-rater reliability analysis (Krippendorff's α = 0.81) across a 3-LLM panel evaluating 12,500 response pairs. Bayesian hierarchical risk modeling for multi-model vulnerability quantification.

Methodology I default to: cross-validation (k-fold, stratified, LOO), Bayesian credible intervals, multiple-testing correction (Bonferroni / FDR / Holm-Šidák), effect-size reporting (Cohen's d, η², Cramer's V), and explicit power analyses. I think of model accuracy as one number among many that has to add up.

Three things I'd add at {{Company}}:

1. **Bayesian fluency that's actually shipped** — PyMC, ArviZ, NumPyro, Stan — with diagnostics (R-hat, ESS, posterior predictive checks) reported as a matter of course, not buried in an appendix.
2. **Comfort across the rigor stack** — From classical ensembles with proper calibration to modern LLM panels with reliability analysis, I pick the right tool rather than the trendy one.
3. **Reproducible deliverables** — Every project ships with a versioned technical report, a model card, and code that runs end-to-end from a clean clone.

Glad to discuss further. Portfolio: [dl1413.github.io/LLM-Portfolio](https://dl1413.github.io/LLM-Portfolio/).

Best,

Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · US work auth · Remote / NYC

---

### Customization notes
- For quant finance: emphasize Bayesian inference, calibration, hypothesis testing. Drop the third project entirely; replace with one sentence on experimental-design coursework.
- For healthcare analytics: lead with Breast Cancer + clinical framing (calibration, sensitivity/specificity tradeoffs at threshold 0.31).
- For experimentation/causal-inference teams: emphasize the "Experimental Design & Causal Inference" coursework line; mention multiple-testing correction and effect-size reporting explicitly.
