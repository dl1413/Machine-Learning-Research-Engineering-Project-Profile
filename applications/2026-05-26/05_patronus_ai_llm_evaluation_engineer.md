# 05 — Patronus AI · LLM Evaluation Engineer

- **Location:** NYC / Remote
- **Family:** Eval Platform
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 Bias Detection (ensemble + Bayesian)
- **JD link:** https://www.patronus.ai/careers (LLM Evaluation Engineer)
- **Resume version:** `resume_v3_eval.pdf`
- **Cover letter:** YES

## JD keywords to echo verbatim (≥3)
1. "LLM evaluation"
2. "hallucination" / "harm detection"
3. "eval harness"
4. "judge model" / "LLM-as-judge"
5. "production"

## Resume bullet stack

**AI Safety Red-Team Evaluation (lead — Eval Platform variant)**
- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while holding Krippendorff α = 0.81
- 47 engineered features across linguistic, semantic, structural axes for jailbreak / refusal-evasion / policy-violation signals

**LLM Textbook Bias Detection (supporting)**
- 67,500 ratings, 4,500 passages, 2.5M tokens, async circuit-breakered API layer, MLflow lineage end-to-end
- Krippendorff α = 0.84, 92% pairwise inter-LLM correlation, Friedman χ² = 42.73, p < 0.001 on 3/5 publishers

## Cover letter

> Hi Patronus team,
>
> I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
> 850 samples/hr, with circuit breakers, async batching, and MLflow
> tracking baked in. Cost per sample landed at $0.018, a 340x reduction
> versus human review. The interesting part for Patronus is the stacking
> layer: a 47-feature meta-classifier that reconciles disagreement between
> the three judges and surfaces per-model blind spots via a PyMC Bayesian
> hierarchical model with 95% HDI per rater. That maps directly to the
> hallucination / harm / policy-violation detection your platform sells.
>
> A second project pushes the same architecture to a different domain:
> 67,500 LLM ratings over 4,500 textbook passages and 2.5M tokens, holding
> Krippendorff α = 0.84 and 92% pairwise correlation, with publisher-level
> bias localized at Friedman χ² = 42.73, p < 0.001. End-to-end MLflow
> lineage, full reproducibility, public technical report.
>
> What I want to bring to Patronus: judge-model engineering, ensemble
> reconciliation, and the rigor scaffolding (inter-rater reliability,
> Bayesian uncertainty, audit trails) that makes eval results trustworthy
> enough to sell as a product.
>
> Available for NYC and remote, 2026 start after my MS Applied Statistics
> at RIT. Code and reports on GitHub (dl1413).
>
> Best,
> Derek Lankeaux
