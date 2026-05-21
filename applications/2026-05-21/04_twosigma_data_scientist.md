# Two Sigma — Data Scientist, Inference (NYC)

- **Role family:** Data Scientist (Bayesian / Causal)
- **Lead project:** P2 — LLM Ensemble Bias Detection (Bayesian inference)
- **Supporting:** P3 — Breast Cancer Classification · P1 — AI Safety Red-Team
- **Source:** twosigma.com/careers (Tier C, NYC Finance ML)
- **Resume version:** `resume_v3_ds` (Projects section reordered → P2 first)

## JD KEYWORDS TO ECHO (fill from live JD before submit)
`__________`, `__________`, `__________`
Bank: Bayesian hierarchical modeling, MCMC, posterior inference, hypothesis
testing, experimental design, partial pooling, credible intervals.

## Resume Projects-section order
1. **LLM Ensemble Bias Detection** — *Data Scientist (Bayesian)* bullets
2. Breast Cancer Classification — *Generalist MLE* bullets (modeling discipline)
3. AI Safety Red-Team Evaluation — *ML Research Engineer* bullets (uncertainty)

## Cover letter

Dear Two Sigma Hiring Team,

Built a multi-LLM bias-rating pipeline that processed 67,500 ratings over 4,500
passages (2.5M tokens) at Krippendorff's alpha = 0.84, and found statistically
significant publisher bias (Friedman chi-squared = 42.73, p < 0.001) in 3 of 5
publishers.

One project I'd point to for an inference role is that LLM-ensemble
bias-detection study: GPT-4o / Claude-3.5 / Llama-3.2, 67,500 ratings. The
headline finding only holds because the pipeline was built to defend it —
Krippendorff's alpha = 0.84 across raters, 92% pairwise correlation, and a PyMC
partial-pooling hierarchical model with R-hat < 1.01 producing 95% HDI
credible intervals per publisher, plus an omnibus Friedman test and post-hoc
Nemenyi comparisons to localize effects. That Bayesian-first, quantify-the-
uncertainty habit is what I'd bring to Two Sigma's inference work.

My other two projects show the modeling discipline transfers. A breast-cancer
classifier picked by evidence — an 8-algorithm benchmark under cross-validation
— reached 99.12% accuracy and ROC-AUC 0.9987 with calibrated thresholds. And an
AI safety red-team eval framework (96.8% accuracy, 95% HDI risk intervals via a
hierarchical model) shows the same instinct on noisy, high-dimensional signal.

I'm based in / available for New York City, targeting a 2026 start once I wrap
my MS in Applied Statistics at RIT. Portfolio, code, and three technical
reports are on my GitHub (dl1413). Happy to walk through the Bayesian pipeline
in detail.

Best,
Derek Lankeaux

## Pre-submit checklist
- [ ] 3+ JD phrases echoed in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio links live
- [ ] Work auth: Authorized to work in the US
- [ ] Salary: open, targeting market for role/location
- [ ] JD PDF saved in this folder
