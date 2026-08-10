# Cover Letter - Two Sigma, Data Scientist, Campus Full-Time

**Location:** New York, NY
**Lead project:** LLM Ensemble Bias Detection (Bayesian) + WBCD Calibration
**JD link:** https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Data-Scientist-Campus-Full-Time/13662

---

Dear Two Sigma Data Science team,

Two Sigma looks for people who can explain *why* they chose an approach
and *what could go wrong* with it, not just execute one. That's the habit
I've built across three published projects - and the reason my strongest
signal for this role is a Bayesian hierarchical study rather than a
single-headline model.

**The Bayesian project.** I ran a multi-LLM bias-detection study using
GPT-4o, Claude-3.5, and Llama-3.2 as raters: 4,500 textbook passages,
2.5M tokens, 67,500 individual ratings. The headline is that 3 of 5
publishers show statistically significant directional bias (Friedman
chi-squared = 42.73, p < 0.001, with Nemenyi post-hoc localizing the
pairs). What makes that result usable is what sits underneath it:

- **PyMC hierarchical model with partial pooling** across publishers so
  small-sample publishers borrow strength from the pool - closer to a
  factor-model prior than a naive per-publisher regression.
- **MCMC diagnostics:** R-hat < 1.01, ESS > 1000, prior/posterior
  predictive checks, and sensitivity analysis on the prior scale.
- **95% HDI credible intervals per publisher and per topic**, so the
  claim is "this publisher is likely biased in this range," not "this
  publisher is biased" (point estimate).
- **Inter-rater reliability grounding.** Krippendorff's alpha = 0.84
  with 92% pairwise correlation across three frontier LLMs - the
  measurement layer before any modeling starts.

**A second data point on model discipline.** A clinical-grade
Wisconsin-Breast-Cancer classifier: 8-algorithm benchmark under nested
CV, Platt calibration cutting ECE by 71.5%, context-specific thresholds
(100% sensitivity at 0.31 for screening), SHAP for feature attribution.
The point isn't the 99.12% accuracy; it's that calibration, threshold
policy, and explanations matter more than the top-line number when the
decision has consequences.

**Why Two Sigma specifically.**

- MS in Applied Statistics at RIT (2026), specialization in Bayesian
  Methods, Experimental Design, and Statistical Learning Theory.
- Comfortable owning the full DS workflow - framing the question,
  writing the SQL, building the model, quantifying uncertainty, and
  communicating impact - which is how DS operates inside a quant firm.
- Familiar with time-series concepts, hypothesis testing, and
  bias-variance tradeoff at the depth Two Sigma expects (I've spent as
  much time defending priors and effect sizes as fitting models).
- Based in / available for New York City.

Portfolio, code, and three technical reports are on GitHub at dl1413.
Happy to walk through the hierarchical model - the PyMC posterior
plots and the sensitivity analysis are probably the most direct
"here's how I think" artifact I have.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
