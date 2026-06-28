# AI Safety Red-Team Evaluation — Bullet Variants

Project header line (use as-is on all framings):
> **AI Safety Red-Team Evaluation Framework** — Independent Research · January 2026
> Tech: GPT-4o, Claude-3.5, Llama-3.2, XGBoost, Stacking Classifier, PyMC, SHAP, MLflow

---

## Research / Frontier-lab framing

**Short (2 bullets)**
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2 → Stacking Classifier) achieving **96.8% accuracy** on automated harm detection across 12,500 response pairs and 6 harm categories.
- Achieved **Krippendorff's α = 0.81** inter-rater reliability and Bayesian hierarchical risk quantification with 95% HDI; published technical report with IEEE 2830-2025 compliance.

**Medium (3–4 bullets)**
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2 → Stacking Classifier) achieving **96.8% accuracy** on automated harm detection across 12,500 response pairs spanning 6 harm categories (precision 97.2%, recall 96.1%, ROC-AUC 0.9923).
- Reached **Krippendorff's α = 0.81** inter-rater reliability across the LLM panel — meeting published thresholds for excellent agreement in human-rater studies.
- Designed **MITRE ATLAS–aligned attack taxonomy** (8 categories) identifying multi-turn escalation as highest-risk vector (31.8% success rate); measured dual-filter defense reducing harm rate 21.8% → 4.8%.
- Implemented Bayesian hierarchical risk model (PyMC) quantifying per-model vulnerability with 95% HDI; full SHAP audit trails and IEEE 2830-2025 / ISO/IEC 23894:2025 compliance.

**Long (5–6 bullets)** — adds:
- Engineered **47 features** capturing linguistic, semantic, and structural harm signals across response pairs.
- Sustained **850 samples/hour** production throughput with circuit breakers, exponential backoff, adaptive rate limiting, and MLflow experiment tracking.

---

## AI Safety / Alignment framing

**Short**
- Built end-to-end automated red-team evaluation system: dual-stage LLM ensemble → ML classifier, **96.8% accuracy**, **340× cost reduction** vs human raters ($0.018/sample), α = 0.81 reliability.
- Developed **MITRE ATLAS–aligned 8-category attack taxonomy**; quantified dual-filter defense reducing harm rate from 21.8% to 4.8% (78% reduction).

**Medium**
- Built end-to-end automated red-team evaluation: GPT-4o + Claude-3.5 + Llama-3.2 ensemble feeds a Stacking Classifier over 47 engineered features, evaluating 12,500 response pairs across 6 harm categories (**96.8% accuracy**, α = 0.81).
- Achieved **340× cost reduction** vs human annotation ($0.018/sample vs $6.12) at 850 samples/hr production throughput.
- Developed **MITRE ATLAS–aligned attack taxonomy** identifying multi-turn escalation as highest-risk vector (31.8%); measured dual-filter defense reducing harm rate from 21.8% → 4.8%.
- Bayesian hierarchical risk modeling with 95% HDI; SHAP audit trails for IEEE 2830-2025 / EU AI Act compliance.

**Long** — adds:
- Designed 6-category harm taxonomy (dangerous information, hate, deception, privacy violation, illegal activity, self-harm) for content classification.
- Published technical report v2.0.0 documenting methodology, reliability analysis, and reproducibility checklist.

---

## Applied / Production ML framing

**Short**
- Shipped production LLM evaluation pipeline at **850 samples/hr** with circuit breakers and MLflow tracking; **96.8% accuracy** on harm detection across 12,500 pairs.
- Reduced annotation cost **340×** vs human baseline ($0.018/sample) while maintaining α = 0.81 reliability.

**Medium**
- Designed and deployed dual-stage LLM evaluation pipeline (GPT-4o, Claude-3.5, Llama-3.2 → Stacking Classifier) achieving **96.8% accuracy** at 850 samples/hr production throughput.
- Engineered fault tolerance: circuit breakers, exponential backoff, adaptive rate limiting, structured logging, MLflow experiment tracking — sustained 80K+ API calls with no manual intervention.
- Cut per-sample evaluation cost by **340×** ($6.12 → $0.018) while preserving α = 0.81 inter-rater reliability — within published thresholds for production use.
- SHAP-based explainability layer exposed via API for stakeholder review and audit trail.

**Long** — adds:
- Built 47-feature engineering pipeline capturing linguistic, semantic, and structural signals.
- Versioned artifacts, model cards, and reproducibility checklist per IEEE 2830-2025.

---

## ML Platform / Infra framing

**Short**
- Engineered LLM-ensemble pipeline sustaining **850 samples/hr** through circuit breakers, exponential backoff, and adaptive rate limiting across three LLM providers; full MLflow tracking and SHAP audit trails.

**Medium**
- Built fault-tolerant multi-provider LLM annotation pipeline (GPT-4o, Claude-3.5, Llama-3.2) handling **80K+ API calls** with circuit breakers, exponential backoff, adaptive rate limiting, and structured logging — sustained 850 samples/hr without manual intervention.
- Instrumented end-to-end MLflow experiment tracking with versioned artifacts, model registry, and reproducibility metadata (random seeds, environment pinning).
- Exposed SHAP explanations as production endpoints alongside Stacking Classifier predictions for downstream audit consumption.
- Documented system for IEEE 2830-2025 (Transparent ML) and ISO/IEC 23894:2025 (AI Risk Management) compliance.

---

## Data Scientist / Quant / Stats framing

**Short**
- Reported **Krippendorff's α = 0.81** inter-rater reliability across 3-LLM panel evaluating 12,500 response pairs; Bayesian hierarchical risk model with 95% HDI uncertainty quantification.

**Medium**
- Conducted inter-rater reliability analysis (Krippendorff's α = 0.81) across a 3-LLM evaluation panel for 12,500 response pairs spanning 6 categories.
- Built **PyMC hierarchical Bayesian model** with partial pooling for per-model risk quantification; reported 95% HDI credible intervals rather than point estimates.
- Validated classifier with stratified k-fold cross-validation; ROC-AUC 0.9923, precision 97.2%, recall 96.1% on held-out data.
- Quantified defense effectiveness with paired comparisons: harm rate dropped 21.8% → 4.8% (78% relative reduction).

---

## Civic / Government / Cross-source data framing

> Use this framing for the **NYC OCCECE Data Manager** role and similar civic-data positions.

**Short**
- Built reproducible data pipeline integrating annotations from **3 independent providers** with documented provenance, anomaly detection, and version-controlled data dictionaries.
- Translated inferential statistical findings (95% HDI intervals, inter-rater reliability α = 0.81) into a published technical report for cross-functional readers.

**Medium**
- Integrated and documented annotations from 3 separate data providers (GPT-4o, Claude-3.5, Llama-3.2) across 12,500 evaluation pairs — maintained data dictionaries, provenance metadata, and a reproducibility checklist as schemas evolved.
- Built anomaly detection and source-followup loop: structured logging, circuit breakers, and automated flagging when expected data was overdue or out of range across 80K+ records.
- Reported descriptive and inferential analyses (inter-rater reliability α = 0.81, Bayesian credible intervals, effect sizes) translated for non-technical readers in a published technical report.
- Maintained MLflow-based monitoring dashboard tracking pipeline KPIs (throughput, error rates, reliability coefficients) over time.
