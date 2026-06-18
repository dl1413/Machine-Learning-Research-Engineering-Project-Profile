# Two Sigma — Quantitative Data Scientist

**Geo:** New York City (on-site)
**Role family:** Data Scientist (Bayesian / Causal)
**Lead project:** P2 — LLM Ensemble Textbook Bias Detection
**Supporting:** P1 (eval rigor), P3 (modeling discipline)
**JD link:** <PASTE_JD_URL_AT_SUBMIT>

---

## Resume re-order

1. **LLM Ensemble Textbook Bias Detection** — use "Data Scientist (Bayesian /
   Causal)" bullets
2. **Clinical-Grade Breast Cancer ML** — Generalist MLE bullets (highlight
   nested CV, calibration, benchmarking)
3. **AI Safety Red-Team Evaluation** — Research Engineer bullets

## JD phrases to echo (≥3)

- "Bayesian" / "MCMC" / "hierarchical"
- "hypothesis testing" / "experimental design"
- "production research code"

## Cover letter

> Dear Two Sigma Research team,
>
> The project I'd point to first is a Bayesian-first LLM bias-detection
> study I ran last quarter: 4,500 textbook passages, 2.5M tokens, 67,500
> LLM ratings from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding —
> that **3 of 5 publishers showed statistically significant directional
> bias (Friedman χ² = 42.73, p < 0.001)** — only holds because the
> pipeline was built to defend it: Krippendorff α = 0.84 across raters,
> 92% pairwise correlation, and a PyMC partial-pooling hierarchical model
> with **R-hat < 1.01, ESS > 1000** producing 95% HDI credible intervals
> per publisher. Post-hoc Nemenyi pairwise comparisons localized the
> effects. That habit — answer with intervals not points, and prove the
> model converged before you trust it — is what I'd want to bring to
> Two Sigma's research stack.
>
> Two related projects: an AI Safety red-team eval (96.8% accuracy, ROC-AUC
> 0.9923, 340x cheaper than human annotation) and a clinical breast-cancer
> classifier (99.12% accuracy, 100% precision, <100ms p95 FastAPI). Both
> were benchmarked against multiple algorithms under nested CV before
> shipping — same evidence-over-fashion approach.
>
> I'm based in / available for New York City, targeting a 2026 start once
> I finish my MS in Applied Statistics at RIT. Code and full technical
> reports on GitHub (dl1413).
>
> Best,
> Derek Lankeaux
