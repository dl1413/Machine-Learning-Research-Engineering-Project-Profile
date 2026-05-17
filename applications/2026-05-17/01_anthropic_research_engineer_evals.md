# 01 — Anthropic · Research Engineer, Evaluations

**Location:** Remote (US) / San Francisco / NYC presence
**Tier:** A — Frontier lab
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting:** LLM Bias Detection (multi-judge rigor), Breast Cancer (production discipline)
**JD source:** anthropic.com/careers — filter "Evaluations" / "Model Safety"
**Resume version to send:** `resume_v_safety.pdf` (Projects section reordered with Red-Team on top)

---

## Tailored Projects section (paste over top of resume Projects block)

### AI Safety Red-Team Evaluation Framework — *Independent Research, Jan 2026*
- Stacked GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges into an XGBoost meta-classifier reaching **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923** on 12,500 response pairs across 6 harm categories
- Cut human-eval cost **340x ($6.12 → $0.018 / sample)** while holding inter-rater reliability at **Krippendorff's alpha = 0.81**
- Engineered 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Modeled multi-judge uncertainty with a PyMC Bayesian hierarchy, producing 95% HDI risk intervals per judge family
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explanations and full provenance trails; eval harness sustains **850 samples/hr** with circuit breakers, exponential backoff, and MLflow run tracking

### LLM Ensemble Textbook Bias Detection — *Independent Research, 2025*
- 67,500 LLM ratings across 4,500 passages (2.5M tokens) at Krippendorff's alpha = 0.84 and 92% pairwise correlation across the same 3-judge ensemble
- PyMC partial-pooling hierarchical model (R-hat < 1.01, ESS > 1000); Friedman chi-squared = 42.73, p < 0.001 localized to 3 of 5 publishers

### Clinical-Grade Breast Cancer Classification — *Independent Research, 2025*
- 8-algorithm benchmark; winning stacking ensemble at 99.12% accuracy, 100% precision, ROC-AUC 0.9987; FastAPI service at <100ms p95 with MLflow registry

---

## Cover letter

> Dear Anthropic Evaluations team,
>
> My most relevant work for Anthropic's mission is an independent AI Safety Red-Team Evaluation framework I built and published in January 2026. It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking classifier on 47 harm-signal features, reaching **96.8% accuracy and ROC-AUC 0.9923** against a 12,500-pair benchmark across 6 harm categories. The pipeline runs at **850 samples/hr for $0.018/sample — a 340x cost reduction versus human annotation** — while holding inter-rater reliability at Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals per judge, and shipped the whole thing under IEEE 2830-2025 audit-trail requirements.
>
> Two other projects round out the eval profile: a 67,500-rating LLM-ensemble bias study (Krippendorff's alpha = 0.84, Friedman chi-squared = 42.73, p < 0.001) and a clinical-grade ML classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987) productionized behind FastAPI at <100ms p95. Together they're the same habit — eval rigor plus production throughput — applied across safety, social-science, and clinical domains.
>
> I'd love to bring that to Anthropic's evaluations work, especially anything touching automated red-teaming, judge-disagreement modeling, or eval-harness throughput. I'm based in / available for New York City and open to remote, targeting a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio, code, and the three technical reports are on my GitHub (dl1413) — the red-team eval is probably the fastest way to see how I think about your problem.
>
> Best,
> Derek Lankeaux

---

## JD-keyword echo plan (fill in after reading the live JD)
- Phrase 1: ________________________
- Phrase 2: ________________________
- Phrase 3: ________________________
