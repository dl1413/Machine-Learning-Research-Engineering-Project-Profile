# Cover Letter — Apple
**Role:** Senior Data Scientist, Experimentation & Causal Inference (Apple Services)
**Location:** New York, NY
**Source:** tealhq.com — Apple Senior Data Scientist, Experimentation & Causal Inference
**Anchor Projects:** LLM Ensemble Bias Detection (Bayesian rigor) · Clinical-Grade Breast Cancer ML (calibration + decision policy)

---

Dear Apple Services Experimentation Team,

I'm applying for the Senior Data Scientist, Experimentation & Causal Inference role. My Applied Statistics MS is anchored in the exact toolkit the posting asks for — **experimental design, causal inference, Bayesian modeling, power analysis, measurement strategy** — and I've applied it end-to-end in three production-shaped projects.

**Bayesian hierarchical modeling on real data.** My LLM Bias Detection project analyzed **67,500 ratings across 4,500 passages, 5 publishers, and multiple topics** using a partial-pooling PyMC hierarchical model. **MCMC diagnostics: R-hat < 1.01, ESS in the expected range.** Cross-group comparisons used **Friedman χ² = 42.73 (p < 0.001)** with **Bonferroni / FDR correction** and **95% HDI credible intervals** — the same measurement-strategy pattern that lets Services distinguish real cross-cohort effects from multiple-comparisons noise. I also handled **inter-rater reliability (Krippendorff's α = 0.84)** and **bootstrap CIs at the passage level**, flagging **12.3% high-uncertainty units** for human review — the analog of holding an experiment open for more power vs. calling it now.

**Calibration and decision policy — the operational side of causal work.** In the Breast Cancer Classification project I hit **99.12% accuracy, ROC-AUC 0.9987**, then Platt-calibrated probabilities (**ECE 71.5% lower: 0.0312 → 0.0089**) and swept thresholds to a **context-adaptive decision policy** (100% sensitivity at 0.31 for screening). That's the same "measure, calibrate, translate to a decision rule" chain that Services experiments need.

**Ready for Apple's stack.** SQL, Python (Pandas/Polars), R, PyMC/Stan, MLflow. Model cards and reproducibility artifacts standard. Comfortable owning eval design, sample sizing, and stakeholder readouts.

I'd welcome the chance to talk about how this fits Apple Services' measurement roadmap.

Best regards,
**Derek Lankeaux, MS**
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
