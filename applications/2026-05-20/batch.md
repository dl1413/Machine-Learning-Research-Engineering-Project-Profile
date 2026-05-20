# Daily Application Batch — 2026-05-20 (5 apps, NYC + Remote)

**Owner:** Derek Lankeaux · LinkedIn: linkedin.com/in/derek-lankeaux · GitHub: github.com/dl1413 · Portfolio: dl1413.github.io/LLM-Portfolio

Each package rotates one of the 3 primary projects as the lead and lists the
other two as support. Resume bullets and cover-letter paragraphs are tailored
from `APPLICATION_SNIPPETS.md`. **Action required from you:** open the live
posting, paste the exact req URL into `application_tracker.csv`, paste the
package below, submit, then flip the row's `status` to `submitted`.

> Note: live req URLs are not pre-filled — search each company's careers page
> for the current matching opening, since postings rotate daily.

---

## 1. Anthropic — Research Engineer, Evaluations  · Remote / NYC · AI Safety
**Lead:** Project 1 (AI Safety Red-Team) · **Support:** Project 2 (Bias), Project 3 (rigor)
**Careers:** careers.anthropic.com

**Cover-letter opener (metric hook):**
> My most relevant work for Anthropic's mission is an independent AI Safety
> Red-Team Evaluation framework I built and published in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
> stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and
> ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories. The
> pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost reduction
> versus human annotation — while holding inter-rater reliability at
> Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian hierarchical
> model that produces 95% HDI risk intervals per judge, and shipped it under
> IEEE 2830-2025 audit-trail requirements. I'd love to bring that combination
> of eval rigor and production throughput to Anthropic's evals team.

**Top resume bullets (place first in Projects):**
- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 to $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

**JD keywords to echo verbatim:** red-teaming, jailbreak detection, harm classification, LLM-as-judge, eval harness, Constitutional AI

---

## 2. Patronus AI — LLM Evaluation Engineer · Remote / NYC · Eval Platform
**Lead:** Project 1 (AI Safety Red-Team, eval-infra framing) · **Support:** Project 2 (Bias)
**Careers:** patronus.ai/careers

**Cover-letter opener (metric hook):**
> I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
> 850 samples/hr, with circuit breakers, async batching, and MLflow tracking
> baked in. Cost per sample landed at $0.018, a 340x reduction versus human
> review. The interesting part for Patronus is the stacking layer: a
> 47-feature meta-classifier that reconciles disagreement between the three
> judges and surfaces per-model blind spots via Bayesian hierarchical
> modeling. That maps directly to the eval-tooling problems you're solving.

**Top resume bullets:**
- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family

**JD keywords to echo verbatim:** evaluation harness, LLM-as-judge, model behavior, eval infrastructure, hallucination/harm detection

---

## 3. Two Sigma — Data Scientist (Inference) · NYC · Finance ML / Bayesian DS
**Lead:** Project 2 (LLM Bias, Bayesian framing) · **Support:** Project 3 (Breast Cancer), Project 1 (rigor)
**Careers:** careers.twosigma.com

**Cover-letter opener (metric hook):**
> One project I'd point to is an LLM-ensemble bias-detection study I ran last
> quarter: 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings from
> GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of 5
> publishers showed statistically significant directional bias (Friedman
> chi-squared = 42.73, p < 0.001) — only holds because the pipeline was built
> to defend it: Krippendorff's alpha = 0.84 across raters, 92% pairwise
> correlation, and a PyMC partial-pooling hierarchical model with R-hat < 1.01
> producing 95% HDI credible intervals per publisher. That Bayesian-first
> habit is what I'd want to bring to Two Sigma's inference work.

**Top resume bullets:**
- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000)
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible claims rather than point estimates
- Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize effects

**JD keywords to echo verbatim:** Bayesian inference, MCMC, hierarchical modeling, hypothesis testing, experimental design, uncertainty quantification

---

## 4. Memorial Sloan Kettering — Clinical ML Engineer, Computational Oncology · NYC · Healthcare ML
**Lead:** Project 3 (Breast Cancer) · **Support:** Project 1 (rigor / audit), Project 2 (stats)
**Careers:** careers.mskcc.org

**Cover-letter opener (metric hook):**
> The work most relevant to MSK is a clinical-grade breast-cancer classifier I
> shipped this year. I benchmarked 8 algorithms end-to-end — Random Forest
> through stacking ensembles — and landed at 99.12% accuracy with 100%
> precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987,
> comfortably above the 90-95% range typically cited for human expert reads.
> Just as important for clinical deployment: the pipeline ships with SHAP
> explanations per prediction, VIF-pruned features, and a FastAPI service
> under 100ms p95, all aligned with IEEE 2830-2025 transparency standards.
> I'd love to apply the same rigor to MSK's computational-oncology work.

**Top resume bullets:**
- Built clinical-grade classifier exceeding the 90-95% human-expert range, with zero false positives across the held-out test set
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation

**JD keywords to echo verbatim:** clinical decision support, diagnostic classification, explainability/SHAP, model validation, healthcare AI

---

## 5. Hugging Face — Research / Evaluation Engineer · Remote · ML Research Engineer
**Lead:** Project 1 (AI Safety Red-Team, research framing) · **Support:** Project 2, Project 3
**Careers:** huggingface.co/jobs

**Cover-letter opener (metric hook):**
> My most relevant work for Hugging Face is a published AI Safety Red-Team
> Evaluation framework: a stacking classifier trained on 47 harm-signal
> features over a 12,500-pair benchmark, reaching 96.8% accuracy and ROC-AUC
> 0.9923 across 6 harm categories, with a PyMC Bayesian hierarchy producing
> 95% HDI risk intervals per judge. The full pipeline — GPT-4o / Claude-3.5 /
> Llama-3.2 ensemble, MLflow lineage, reproducible report — is on my GitHub.
> I build evals the way HF ships them: open, reproducible, and benchmarked.

**Top resume bullets:**
- Trained stacking classifier on 47 engineered features for harm detection; 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923 on 12,500-sample benchmark
- Modeled multi-judge uncertainty via PyMC Bayesian hierarchy, producing 95% HDI risk intervals for downstream policy decisions
- Documented results in research-grade technical report; reproducible pipeline released alongside published findings

**JD keywords to echo verbatim:** research engineer, reproducible evaluation, benchmarking, open-source, LLM-as-judge

---

## Universal closer (append to every cover letter)
> I'm based in / available for New York City and open to remote, targeting a
> 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio, code,
> and the three technical reports referenced above are on my GitHub (dl1413).
> Happy to walk through any of them.
>
> Best,
> Derek Lankeaux

## Pre-submit checklist (per app)
- [ ] Lead project sits at top of Projects section
- [ ] Cover letter opens with a metric hook
- [ ] 3+ JD phrases appear verbatim in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio URLs live
- [ ] JD-specific framework (PyTorch/JAX/Ray/vLLM) added to skills if listed
- [ ] Salary + work-authorization fields answered
- [ ] Live req URL pasted into application_tracker.csv, status flipped to submitted
