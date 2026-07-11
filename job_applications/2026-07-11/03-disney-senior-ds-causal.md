# Disney DTC — Senior Data Scientist, Experimentation & Causal Inference

**Location:** New York, NY (Disney+, Hulu, ESPN Direct-to-Consumer)
**Posting:** https://jobs.disneycareers.com/job/new-york/senior-data-scientist-experimentation-and-causal-inference/391/89132547520
**Lead project:** LLM Ensemble Bias Detection (for Bayesian/statistical rigor) + cross-project experimentation credibility

---

## Cover Letter

Hi Disney DTC Data Science team,

The Senior Data Scientist role architecting experiments across Disney+, Hulu, and ESPN is a strong match for how I've been working — designing measurement that survives multiple-comparison, multi-model, and multi-population scrutiny.

**Experimentation & Bayesian inference at scale.** My **LLM Ensemble Textbook Bias Detection** system processed **67,500 ratings across 4,500 passages** and used a PyMC hierarchical model with partial pooling — MCMC R-hat < 1.01, 95% HDI, and Bonferroni/FDR correction — to detect **credible publisher-level differences (3/5 significant, Friedman χ² = 42.73, p < 0.001)**. Spearman inter-publisher correlations up to 0.74 revealed structural editorial relationships; a cross-topic heatmap showed Social Issues driving the largest polarization (Δ = 1.36). That's the same toolkit a subscriber-journey experimentation team applies to multi-treatment A/B/n, cluster-correlated units, and heterogeneous treatment effects.

**Calibration and decision policy.** In the **Breast Cancer ML** system (99.12% accuracy, ROC-AUC 0.9987), I applied Platt scaling to bring ECE from 0.0312 down to 0.0089 (71.5% reduction) and built context-adaptive thresholds — 100% sensitivity at 0.31 for screening, high-precision at 0.67 for diagnostic. That translates directly to policy-level threshold choices in engagement, retention, and offer-targeting experiments.

**LLM-driven measurement.** My **AI Safety Red-Team Evaluation** shows I can wire LLM-as-judge and human-in-the-loop labeling into a production evaluation pipeline (340× cost reduction, α = 0.81 reliability), which increasingly matters for measuring subjective outcomes — content-quality, ad relevance, safety — at scale.

Applied Statistics MS at RIT (expected 2026), SQL-fluent, comfortable owning the full loop from problem framing to stakeholder readout.

Thanks for considering my application,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Attach

- Resume PDF
- `LLM_Bias_Detection_Publication.pdf`
- `Breast_Cancer_Classification_Publication.pdf`
