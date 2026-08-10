# Cover Letter — AI Safety / Responsible AI Engineer

**Placeholders to replace:** `[Company]`, `[Team/Product]`, `[JD hook]`.

---

Dear [Company] Hiring Team,

I'm writing about the [Team/Product] role. What caught my eye was [JD hook —
e.g., "shipping evidence-grade safety artifacts alongside models"] — I've built
the pipeline that produces exactly those artifacts.

The core project is an **AI Safety Red-Team Evaluation Framework**: a dual-stage
LLM ensemble that runs GPT-4o, Claude-3.5, and Llama-3.2 over 12,500 response
pairs across six harm categories, achieving **96.8% accuracy** with
**Krippendorff's alpha = 0.81** (audit-grade IRR) at **340x lower cost than
human annotation ($0.018/sample vs. $6.12)**. It ships with 47 engineered
harm-signal features, a Bayesian hierarchical model (95% HDI, R-hat < 1.01) for
multi-model risk, SHAP explanations, and a complete audit trail aligned to
**IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI Act**. The technical report
is written the way a compliance reviewer would read it, not just an ML
researcher.

Two supporting projects show the same rigor transfers. My **LLM Ensemble Bias
Detection** graded 67,500 ratings across 4,500 textbook passages and surfaced
publisher-level bias with p < 0.001 and 92% pairwise LLM correlation. My
**Clinical-Grade Classifier** hits 99.12% accuracy with zero false positives
and <100ms p95 FastAPI serving — the "high-stakes decision + SHAP audit + model
card" pattern that safety work rests on.

I'd like to bring the same discipline — ensembles, IRR, hierarchical
uncertainty, standards-aligned reporting — to [Team/Product]. Portfolio and
reports linked below.

Best,
Derek Lankeaux
