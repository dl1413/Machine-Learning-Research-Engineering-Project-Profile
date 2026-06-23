# Application 5 — Patronus AI · LLM Evaluation Engineer

| Field | Value |
|---|---|
| Date | 2026-05-25 (Mon) |
| Company | Patronus AI |
| Role | LLM Evaluation Engineer |
| Role family | Eval platform / Eval Infra |
| Location | Remote (US) |
| Source | patronus.ai/careers |
| Lead project | Project 1 — AI Safety Red-Team Evaluation Framework |
| Supporting | Project 2 — LLM Bias Detection (multi-LLM at production scale) |
| Resume version | resume_v3_eval.pdf |
| Cover letter | Yes |
| Referral | No |

---

## Tailored resume bullets

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Built production LLM eval harness processing 850 samples/hr with circuit breakers, exponential backoff, async API integration, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a 47-feature meta-classifier reaching 96.8% agreement with gold human labels on 12,500 pairs
- Cut eval cost 340x ($6.12 to $0.018/sample) while holding Krippendorff's alpha = 0.81 across the three judges
- Quantified judge disagreement with PyMC Bayesian hierarchy producing 95% HDI per harm category — surfaces per-model blind spots, not just aggregate scores

**LLM Ensemble Bias Detection** — *Independent Research, Jan 2026*
- Operated the same 3-LLM ensemble at 67,500 ratings, 2.5M tokens, 4,500 passages with end-to-end MLflow lineage and circuit-breakered async API integration
- Surfaced publisher bias at Friedman chi-squared = 42.73, p < 0.001 with alpha = 0.84 reliability scaffolding

## Cover letter

Dear Patronus AI Team,

I'm applying because I've spent the last few months building exactly the kind
of multi-LLM evaluation infrastructure your product abstracts — and I'd
rather help build that as a product than keep rebuilding it project by
project.

The headline project is an AI Safety Red-Team eval harness that ensembles
GPT-4o, Claude-3.5, and Llama-3.2 as judges and trains a 47-feature stacking
classifier on a 12,500-pair benchmark across 6 harm categories. It hits
**96.8% accuracy / ROC-AUC 0.9923** at **850 samples/hr and $0.018/sample —
a 340x cost reduction** versus human annotation, with Krippendorff's
alpha = 0.81 to defend the labels. The interesting layer for Patronus is the
**Bayesian hierarchical model** that reconciles judge disagreement and
surfaces per-model blind spots at 95% HDI — point scores aren't enough when
customers need to know *why* their eval flagged something.

The same stack ran at 67,500 ratings across 2.5M tokens for a publisher-bias
study (alpha = 0.84, Friedman p < 0.001) — circuit breakers, exponential
backoff, async batching, and MLflow lineage held end-to-end, which is what
turns a notebook eval into a platform eval.

I'm finishing my MS in Applied Statistics at RIT (2026) and open to remote.
Excited to talk.

Best,
Derek Lankeaux

## JD keyword echoes

- "LLM evaluation"
- "eval harness"
- "LLM-as-judge"
- "hallucination" / "harm detection"
- "production"
- "async" / "batching"

## Quality checklist

- [x] Lead project at top (Red-Team)
- [x] Metric hook (340x + 96.8% + 850 samples/hr)
- [x] 3+ JD phrases verbatim
- [x] Remote availability stated
- [x] Salary: "open, targeting market for remote eval-engineer"
- [x] Work auth: US authorized
