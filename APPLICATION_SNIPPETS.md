# Application Snippets — Project-Keyed, Paste-Ready

Three projects, three sets of snippets each: **resume bullets**, **one-liner
hook**, **cover letter paragraph**. Pick the version that matches the role
family (Safety / MLE / Research / Healthcare / DS).

---

## Project 1 — AI Safety Red-Team Evaluation Framework

### One-liner hook (cover letter opener)

> *Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that detects
> harmful AI outputs at **96.8% accuracy and 340x lower cost** than human
> annotation, while preserving audit-grade reliability (Krippendorff's
> alpha = 0.81).*

### Resume bullets — AI Safety / Red-Team / Alignment roles

- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 to $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

### Resume bullets — LLM Evaluation / Eval Infra roles

- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family

### Resume bullets — ML Research Engineer / Applied Research

- Trained stacking classifier on 47 engineered features for harm detection; 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923 on 12,500-sample benchmark
- Modeled multi-judge uncertainty via PyMC Bayesian hierarchy, producing 95% HDI risk intervals for downstream policy decisions
- Documented results in research-grade technical report; reproducible pipeline released alongside published findings

### Resume bullets — Data Engineer (ETL/ELT, Pipelines)

- Built Python ELT pipeline ingesting 12,500 records from three external APIs into staged raw/curated layers, with idempotent loads and Git-versioned configs
- Authored SQL transformations (CTEs, window functions) to clean, deduplicate, and join multi-source rating data into an analytics-ready table covering 6 categories and 47 engineered columns
- Hardened ingestion with retry/backoff and circuit breakers, sustaining ~850 records/hour with <0.5% failure rate; tracked runs and artifacts in MLflow for lineage
- Implemented data quality gates (schema validation, null/range checks, inter-rater agreement >= 0.81) before promotion to the curated layer; documented schemas and workflows for analyst consumption

### Cover letter paragraph — AI Safety lab (Anthropic / OpenAI / METR / Apollo)

> My most relevant work for [Company]'s mission is an independent AI Safety
> Red-Team Evaluation framework I built and published in January 2026. It
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
> stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and
> ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories.
> The pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost
> reduction versus human annotation — while holding inter-rater reliability
> at Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian
> hierarchical model that produces 95% HDI risk intervals per judge, and
> shipped the whole thing under IEEE 2830-2025 audit-trail requirements.
> I'd love to bring that combination of eval rigor and production throughput
> to [specific team/product at Company].

### Cover letter paragraph — Eval platform (Patronus / Galileo / Arize)

> I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
> 850 samples/hr, with circuit breakers, async batching, and MLflow tracking
> baked in. Cost per sample landed at $0.018, a 340x reduction versus
> human review. The interesting part for [Company] is the stacking layer:
> a 47-feature meta-classifier that reconciles disagreement between the
> three judges and surfaces per-model blind spots via Bayesian hierarchical
> modeling. That maps directly to the eval-tooling problems you're solving.

---

## Project 2 — LLM Ensemble Textbook Bias Detection System

### One-liner hook

> *Built a multi-LLM bias-rating pipeline that processed **67,500 ratings
> over 4,500 passages (2.5M tokens)** at Krippendorff's alpha = 0.84, and
> found statistically significant publisher bias (Friedman chi-squared =
> 42.73, p < 0.001) in 3 of 5 publishers.*

### Resume bullets — Trust & Safety / Content Policy / Integrity

- Designed multi-LLM bias-rating system covering 4,500 textbook passages and 2.5M tokens; surfaced significant publisher-level bias (p < 0.001) in 3/5 publishers
- Held inter-rater reliability at Krippendorff's alpha = 0.84 and 92% pairwise correlation across GPT-4o, Claude-3.5, Llama-3.2
- Built async API layer with circuit breakers and exponential backoff sustaining 67,500 rated samples without manual intervention

### Resume bullets — Data Scientist (Bayesian / Causal)

- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; achieved MCMC convergence (R-hat < 1.01, ESS > 1000)
- Produced 95% HDI credible intervals per publisher and per topic, enabling defensible "this publisher is biased" claims rather than point estimates
- Ran Friedman omnibus test (chi-squared = 42.73, p < 0.001) plus post-hoc Nemenyi pairwise comparisons to localize effects

### Resume bullets — LLM Eval / Research Engineer

- Operated 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage
- Engineered prompt templates and rubric for bias scoring; validated rubric stability via 92% pairwise inter-LLM correlation
- Published reproducible technical report with methodology, priors, sensitivity analysis, and full posterior visualizations

### Resume bullets — Data Engineer (ETL/ELT, Pipelines)

- Designed and operated an ETL workflow ingesting 4,500 source documents (2.5M tokens), fanning out 67,500 rating calls across 3 external APIs into a normalized relational schema
- Wrote Python extract/load tasks and SQL aggregations (window functions, CTEs, joins) producing publisher/source/category roll-ups for downstream BI dashboards
- Built reliability layer with exponential backoff, rate limiting, and circuit breakers, sustaining 80K+ outbound API calls at <0.5% failure rate
- Enforced data quality via automated agreement checks (Krippendorff's alpha = 0.84, 92% pairwise correlation) and schema validation before promotion to the curated layer; tracked runs in MLflow with Git-versioned configs

### Cover letter paragraph — Data / Stats-heavy role

> One project I'd point to is an LLM-ensemble bias-detection study I ran
> last quarter: 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings
> from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of
> 5 publishers showed statistically significant directional bias (Friedman
> chi-squared = 42.73, p < 0.001) — only holds because the pipeline was
> built to defend it: Krippendorff's alpha = 0.84 across raters, 92%
> pairwise correlation, and a PyMC partial-pooling hierarchical model with
> R-hat < 1.01 producing 95% HDI credible intervals per publisher. That
> Bayesian-first habit is what I'd want to bring to [Company]'s inference
> work.

### Cover letter paragraph — LLM platform / eval startup

> I'm interested in [Company] because I've spent the last few months
> building exactly the kind of multi-LLM evaluation infrastructure your
> product abstracts. My textbook-bias study ran 67,500 ratings through a
> GPT-4o / Claude-3.5 / Llama-3.2 ensemble over 4,500 passages, with
> circuit-breakered async API integration and MLflow lineage end-to-end.
> The eval rubric held Krippendorff's alpha = 0.84 and surfaced
> publisher-level bias at p < 0.001 — the kind of result that needs the
> reliability scaffolding to be trusted. I'd love to help build that
> scaffolding as a product.

---

## Project 3 — Clinical-Grade Breast Cancer ML Classification System

### One-liner hook

> *Trained an 8-algorithm benchmark ensemble that hits **99.12% accuracy,
> 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987**
> on breast-cancer classification — above the 90–95% human-expert range
> — and deployed it as a <100ms p95 FastAPI service.*

### Resume bullets — Applied ML / MLE roles

- Benchmarked 8 algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, +2) and shipped winning ensemble at 99.12% accuracy / ROC-AUC 0.9987
- Applied VIF-based multicollinearity pruning, SMOTE class-balancing, and RFE feature selection to lift recall to 98.59% while holding precision at 100%
- Deployed as a containerized FastAPI service with MLflow model registry; p95 latency under 100ms

### Resume bullets — Healthcare / Clinical ML

- Built clinical-grade classifier exceeding the 90-95% human-expert range, with zero false positives across the held-out test set
- Implemented SHAP-based explanations per prediction to satisfy IEEE 2830-2025 transparency requirements for clinical decision support
- Designed preprocessing for clinical tabular data: VIF multicollinearity diagnostics, SMOTE for class imbalance, stratified cross-validation

### Resume bullets — Generalist MLE / Research Engineer

- Ran rigorous model selection: 8-algorithm benchmark with nested cross-validation, SHAP-based feature attribution, and bias / fairness audit
- Achieved 100% precision and 98.59% recall on a high-stakes binary classification task; documented full pipeline reproducibly
- Productionized winning model behind FastAPI with MLflow model registry and <100ms p95 latency

### Resume bullets — Data Engineer (ETL/ELT, Pipelines)

- Built a reproducible Python/SQL data pipeline transforming raw clinical tabular records into a feature-ready warehouse table, with logged data-quality metrics at every step
- Implemented preprocessing stages -- type coercion, null handling, VIF-based multicollinearity pruning, SMOTE rebalancing, RFE feature selection -- and exported curated views for analyst/BI consumption
- Packaged the workflow in Docker, exposed it via FastAPI (<100ms p95), and versioned datasets and models in MLflow registry for end-to-end lineage
- Documented schemas, transformations, and run procedures in Git so analysts and engineers could rerun and extend the pipeline independently

### Cover letter paragraph — Healthcare ML role

> The work most relevant to [Company] is a clinical-grade breast-cancer
> classifier I shipped this year. I benchmarked 8 algorithms end-to-end —
> Random Forest through stacking ensembles — and landed at 99.12% accuracy
> with 100% precision (zero false positives), 98.59% recall, and ROC-AUC
> 0.9987, comfortably above the 90-95% range typically cited for human
> expert reads. Just as important for clinical deployment: the pipeline
> ships with SHAP explanations per prediction, VIF-pruned features, and a
> FastAPI service under 100ms p95, all aligned with IEEE 2830-2025
> transparency standards. I'd love to apply the same rigor to
> [Company's specific clinical-AI problem].

### Cover letter paragraph — Generalist MLE role

> A recent project that captures how I work: I built a clinical-grade
> ensemble classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987)
> by benchmarking 8 algorithms — Random Forest, XGBoost, LightGBM,
> AdaBoost, Stacking, Voting — under nested cross-validation, then
> productionized the winner behind a FastAPI service at <100ms p95. The
> habits I want to bring to [Company]: pick models by evidence not
> fashion, ship the full pipeline (preprocessing, SHAP explanations,
> MLflow registry), and treat latency as a feature.

### Cover letter paragraph — Entry-Level Data Engineer (Mayzax)

> Across three independent projects I've built the kind of pipelines
> Mayzax's clients rely on: Python + SQL ETL/ELT flows that ingest from
> external APIs, land raw payloads to a staging zone, transform with SQL
> (CTEs, window functions, joins) into curated analytics tables, and
> hand off BI-ready datasets to downstream consumers. The largest run
> moved 80K+ API calls and 2.5M tokens through circuit breakers,
> exponential backoff, and schema/data-quality gates with <0.5%
> failure. Everything is versioned in Git with MLflow-tracked runs for
> lineage. I'd be excited to bring that pipeline discipline to Mayzax
> and grow it onto a modern cloud stack (AWS/Azure/GCP, Airflow, dbt,
> Spark) under your senior mentors.

---

## Universal Cover Letter Closer

> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
> code, and the three technical reports referenced above are on my GitHub
> (dl1413). Happy to walk through any of them — the red-team eval is
> probably the fastest way to see how I think about [Company's problem].
>
> Best,
> Derek Lankeaux

---

## Quick-Reference: Which Snippet for Which Job?

| If the JD says... | Lead with | Snippet section |
|---|---|---|
| "red-team", "alignment", "jailbreak", "harm", "Constitutional AI" | Project 1 | Safety / Red-Team |
| "evaluation harness", "LLM-as-judge", "model behavior" | Project 1 | LLM Evaluation |
| "Bayesian", "MCMC", "hierarchical", "causal inference" | Project 2 | Data Scientist (Bayesian) |
| "fairness", "bias", "trust & safety", "content policy" | Project 2 | Trust & Safety |
| "clinical", "healthcare", "diagnostic", "EHR" | Project 3 | Healthcare ML |
| "ML engineer", "applied ML", "production model" | Project 3 | Applied ML / MLE |
| "research engineer", "applied research" (generic) | Project 1 | ML Research Engineer |
| "data engineer", "ETL/ELT", "data pipeline", "data warehouse", "Airflow / dbt / Spark / Kafka" | Project 2, then 1, then 3 | Data Engineer (ETL/ELT, Pipelines) |
