# 02 — Hugging Face · LLM Evaluation Research Engineer

| Field | Value |
|---|---|
| **Company** | Hugging Face |
| **Role** | LLM Evaluation Research Engineer (or "Eval / Open LLM Leaderboard") |
| **Location** | Remote (US); NYC office available |
| **Source** | apply.workable.com/huggingface (apply direct) |
| **JD link** | _VERIFY before submit:_ https://huggingface.co/jobs (filter: Evaluation / Research) |
| **Lead project** | P1 — AI Safety Red-Team Evaluation |
| **Supporting** | P2 — LLM Bias Detection (HF datasets-style eval); P3 — Breast Cancer (model selection rigor) |
| **Resume version** | `resume_v3_evals.pdf` |
| **Cover letter** | Yes — see below |
| **Referral** | Check LinkedIn for HF researchers I've engaged with on the Hub |

## JD keyword echo (fill in after reading JD)

- [ ] LLM evaluation / eval harness / lm-eval-harness
- [ ] benchmark / leaderboard
- [ ] inter-annotator / inter-rater
- [ ] [HF-specific phrase: e.g. "open-source", "community", "transformers"]
- [ ] [HF-specific phrase 2]

## Resume bullets (top 4 — paste into "Selected Projects")

From `APPLICATION_SNIPPETS.md` → P1 → "LLM Evaluation / Eval Infra":

- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family
- Operated 3-LLM ensemble at production scale across a 2nd domain: 67,500 ratings, 2.5M tokens, Krippendorff's α = 0.84

## Cover letter draft

Dear Hugging Face Research team,

I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
850 samples/hr, with circuit breakers, async batching, and MLflow tracking
baked in. Cost per sample landed at $0.018, a 340x reduction versus
human review. The interesting part for Hugging Face's eval and leaderboard
work is the stacking layer: a 47-feature meta-classifier that reconciles
disagreement between the three judges and surfaces per-model blind spots
via Bayesian hierarchical modeling (R-hat < 1.01, 95% HDI per judge).
That maps directly to "how do we trust an LLM-as-judge benchmark."

I've now built two production multi-LLM eval pipelines on this template
— the red-team evaluation above and a bias-detection study (67,500
ratings, 4,500 passages, Krippendorff's α = 0.84, Friedman χ² = 42.73,
p < 0.001) — so I'd come in with reusable patterns rather than first
principles. Both publications are open and reproducible, in the spirit
of the Hub.

I'm based in / available for New York City and open to remote, targeting
a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
code, and the three technical reports referenced above are on my GitHub
(dl1413). Happy to walk through any of them.

Best,
Derek Lankeaux
