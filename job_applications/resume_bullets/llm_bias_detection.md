# LLM Ensemble Bias Detection — Bullet Variants

Project header line:
> **LLM Ensemble Textbook Bias Detection System** — Independent Research · January 2026
> Tech: GPT-4o, Claude-3.5, Llama-3.2, PyMC, ArviZ, MLflow, FastAPI, LangChain

---

## Research / Frontier-lab framing

**Short**
- Built multi-LLM bias evaluation framework processing **67,500 ratings across 4,500 textbook passages (2.5M tokens)**; α = 0.84 inter-rater reliability, 92% pairwise correlation.
- PyMC hierarchical Bayesian model (R-hat < 1.01) detected credible publisher-level bias in 3/5 publishers (Friedman χ² = 42.73, **p < 0.001**).

**Medium**
- Built multi-LLM bias evaluation framework (GPT-4o + Claude-3.5 + Llama-3.2) processing **67,500 ratings across 4,500 textbook passages** and 2.5M tokens at production scale.
- Achieved **Krippendorff's α = 0.84** inter-rater reliability with 92% pairwise correlation across the three frontier models.
- Implemented PyMC hierarchical Bayesian model with partial pooling; verified MCMC convergence (R-hat < 1.01, ESS diagnostics) and reported publisher-level credible bias with 95% HDI.
- Identified statistically significant findings (Friedman χ² = 42.73, **p < 0.001**) in 3/5 publishers analyzed.

**Long** — adds:
- Spearman correlation matrix revealed structural editorial relationships across publishers (ρ up to 0.74).
- Cross-topic heatmap surfaced Social Issues as highest polarization (Δ = 1.36 points across publishers).
- Bootstrap passage-level CIs flagged 12.3% of passages as high-uncertainty for human expert review.

---

## AI Safety / Alignment framing

**Short**
- Generalized multi-LLM-panel-with-Bayesian-pooling methodology to bias detection: α = 0.84 reliability, p < 0.001 publisher-level findings across 67,500 ratings.

**Medium**
- Extended LLM-ensemble evaluation methodology beyond harm detection to content bias measurement (GPT-4o + Claude-3.5 + Llama-3.2 over 4,500 passages).
- Reached α = 0.84 inter-rater reliability and 92% pairwise correlation — demonstrating the methodology generalizes across normative evaluation domains.
- Used Bayesian hierarchical partial-pooling to separate publisher-level signal from passage-level noise; verified convergence (R-hat < 1.01) before drawing conclusions.
- Bootstrap uncertainty quantification flagged 12.3% of passages as high-uncertainty — supports human-in-the-loop review workflows analogous to harm-eval pipelines.

---

## Applied / Production ML framing

**Short**
- Deployed multi-LLM evaluation pipeline handling **2.5M tokens** across 3 providers with circuit breakers, exponential backoff, and MLflow experiment tracking.

**Medium**
- Productionized multi-provider LLM evaluation pipeline (GPT-4o, Claude-3.5, Llama-3.2) processing 67,500 ratings on 2.5M tokens with FastAPI orchestration.
- Engineered fault tolerance: circuit breakers, exponential backoff, adaptive rate limiting per provider, structured logging — pipeline survived sustained multi-day runs without manual intervention.
- Full MLflow experiment tracking with versioned artifacts, model cards, and reproducible random seeds.
- LangChain-based prompt management with versioned prompt templates and A/B-able prompt variants.

---

## ML Platform / Infra framing

**Short**
- Built multi-provider LLM orchestration layer (3 vendors, **80K+ API calls**, 2.5M tokens) with circuit breakers, exponential backoff, MLflow tracking, and LangChain prompt versioning.

**Medium**
- Engineered production-grade orchestration for 3 LLM providers (GPT-4o, Claude-3.5, Llama-3.2) handling 67,500 evaluation calls with provider-specific rate-limit handling and graceful degradation.
- Built MLflow experiment tracking layer with versioned prompts (LangChain), input fingerprints, and output provenance — supports replay and audit of any single annotation.
- FastAPI service layer exposes evaluation endpoints with calibrated uncertainty estimates from the underlying PyMC model.

---

## Data Scientist / Quant / Stats framing

**Short**
- PyMC hierarchical Bayesian model with partial pooling; verified MCMC convergence (R-hat < 1.01, ESS) and reported publisher-level credible bias with 95% HDI across 4,500 passages.
- Friedman χ² = 42.73, **p < 0.001** for cross-publisher comparison; Spearman correlation ρ up to 0.74 revealing structural editorial relationships.

**Medium**
- Built PyMC hierarchical Bayesian model with partial pooling across 5 publishers and 4,500 passages; verified MCMC convergence (R-hat < 1.01) and effective sample sizes before posterior inference.
- Reported **95% HDI credible intervals** at the passage level; bootstrap analysis flagged 12.3% of passages as high-uncertainty for expert review.
- Applied non-parametric hypothesis testing (Friedman χ² = 42.73, p < 0.001) for cross-publisher comparison; reported effect sizes alongside p-values.
- Spearman correlation matrix (ρ up to 0.74) revealed structural editorial relationships across publishers — interpreted with explicit caveats about correlational evidence.
- Inter-rater reliability quantified via Krippendorff's α = 0.84 across the three-model panel.

---

## Civic / Government / Cross-source data framing

> Use this framing for civic-data and policy-analytics roles.

**Short**
- Integrated and analyzed data across **5 publishers and 4,500 passages**; built per-source provenance documentation and uncertainty quantification flagging 12.3% of cases for human review.
- Translated Bayesian and non-parametric inferential findings (R-hat < 1.01, p < 0.001) into a published report written for cross-functional, non-technical readers.

**Medium**
- Integrated and curated content across **5 separate publishers** (4,500 passages, 2.5M tokens), maintaining source-level metadata, provenance, and a documented data dictionary.
- Built a partial-pooling Bayesian model to separate publisher-level signal from noise — the same statistical pattern used to combine data of varying quality across agencies.
- Reported findings with **explicit uncertainty**: 95% HDI intervals at the unit level, bootstrap CIs flagging 12.3% of items as high-uncertainty for expert follow-up rather than presenting all results with equal confidence.
- Communicated cross-source comparison findings (Friedman χ² = 42.73, p < 0.001) and structural correlations (Spearman ρ up to 0.74) in a publication-quality report for non-technical readers.
