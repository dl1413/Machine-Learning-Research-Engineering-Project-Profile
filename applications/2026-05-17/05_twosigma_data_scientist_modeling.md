# 05 — Two Sigma · Data Scientist, Modeling (Bayesian / Causal)

**Location:** New York City (on-site)
**Tier:** C — NYC Finance
**Lead project:** LLM Ensemble Textbook Bias Detection
**Supporting:** AI Safety Red-Team (production eval), Breast Cancer (rigorous model selection)
**JD source:** twosigma.com/careers — filter "Data Scientist" / "Modeling" / "Quantitative Researcher"
**Resume version to send:** `resume_v_ds_bayesian.pdf`

---

## Tailored Projects section

### LLM Ensemble Bias Detection — *Independent Research, 2025*
- Fit a PyMC Bayesian hierarchical model with partial pooling across publishers; **MCMC convergence at R-hat < 1.01, ESS > 1000**
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible group-level claims rather than point estimates
- Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) with Nemenyi pairwise post-hoc to localize effects to 3 of 5 publishers
- Operated a 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage; held Krippendorff's alpha = 0.84 and 92% pairwise correlation
- Published reproducible technical report with methodology, priors, sensitivity analysis, and full posterior visualizations

### AI Safety Red-Team Evaluation — *Jan 2026*
- 47-feature stacking classifier at 96.8% accuracy, ROC-AUC 0.9923 on 12,500 pairs; 850 samples/hr production harness; PyMC Bayesian hierarchy for multi-judge uncertainty

### Clinical-Grade Breast Cancer Classification — *2025*
- 8-algorithm nested-CV benchmark; 99.12% accuracy, 100% precision, ROC-AUC 0.9987; FastAPI service at <100ms p95

---

## Cover letter

> Dear Two Sigma Modeling team,
>
> One project I'd point to is an LLM-ensemble bias-detection study I ran last quarter: 4,500 textbook passages, 2.5M tokens, **67,500 LLM ratings** from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of 5 publishers showed statistically significant directional bias (**Friedman chi-squared = 42.73, p < 0.001**) — only holds because the pipeline was built to defend it: Krippendorff's alpha = 0.84 across raters, 92% pairwise correlation, and a PyMC partial-pooling hierarchical model with **R-hat < 1.01, ESS > 1000** producing 95% HDI credible intervals per publisher. That Bayesian-first, "report uncertainty not point estimates" habit is what I'd want to bring to Two Sigma.
>
> Two complementary projects show the same discipline at different scales: a 12,500-pair AI Safety Red-Team eval framework (96.8% accuracy, ROC-AUC 0.9923, 850 samples/hr at $0.018 / sample, 340x cheaper than human review) and a clinical-grade classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987) productionized behind a FastAPI service at <100ms p95. Both are end-to-end: framing, design, modeling, uncertainty, and deployment.
>
> I'm finishing an MS in Applied Statistics at RIT (specialization: Bayesian methods, experimental design) and am based in / available for New York City for on-site, 2026 start.
>
> Best,
> Derek Lankeaux

---

## JD-keyword echo plan
- Phrase 1: ________________________
- Phrase 2: ________________________
- Phrase 3: ________________________
