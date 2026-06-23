# 02 — Hugging Face · Research Engineer, Open LLM Evaluation

- **Location:** Remote (US)
- **Family:** LLM Eval / Open Research
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 Bias Detection
- **JD link:** https://apply.workable.com/huggingface/ (Open LLM Eval / Research Engineer)
- **Resume version:** `resume_v3_eval.pdf`
- **Cover letter:** YES

## JD keywords to echo verbatim (≥3)
1. "open-source" / "open models"
2. "evaluation harness"
3. "LLM-as-judge"
4. "benchmark"
5. "reproducibility"

## Resume bullet stack

**AI Safety Red-Team Evaluation Framework (lead — Eval Engineering variant)**
- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
- Quantified judge disagreement with a Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family
- Open-sourced reproducible pipeline alongside published technical report

**LLM Textbook Bias Detection (supporting)**
- 3-LLM ensemble at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage
- Validated rubric stability via 92% pairwise inter-LLM correlation, Krippendorff α = 0.84

## Cover letter

> Hi Hugging Face team,
>
> I'm interested in the Open LLM Evaluation role because I've spent the past
> two quarters building exactly the open-source evaluation harness work your
> team ships into the community. My AI Safety Red-Team framework runs a
> 3-model LLM-as-judge ensemble — GPT-4o, Claude-3.5, Llama-3.2 — against a
> 12,500-pair benchmark across 6 harm categories, hitting 96.8% accuracy
> and ROC-AUC 0.9923. The harness sustains 850 samples/hr with circuit
> breakers, exponential backoff, MLflow lineage, and a stacking meta-classifier
> over 47 harm-signal features. Cost lands at $0.018/sample, a 340x cut
> from human annotation. The whole stack — code, JD-grade technical report,
> and reproducible pipeline — is public.
>
> A second project pushes the same harness in a different direction: 67,500
> LLM ratings across 4,500 textbook passages and 2.5M tokens, with a PyMC
> partial-pooling hierarchical model producing 95% HDI per publisher and
> a Friedman omnibus χ² = 42.73, p < 0.001 localizing publisher-level bias.
> Krippendorff α = 0.84, 92% pairwise inter-LLM correlation.
>
> What I want to bring to Hugging Face is exactly that mix: open-source
> reproducibility, multi-model benchmark rigor, and an obsession with
> inter-rater reliability as the real metric behind "LLM-as-judge."
>
> Available for remote and NYC, 2026 start after my MS Applied Statistics
> at RIT. Code and reports on GitHub (dl1413).
>
> Best,
> Derek Lankeaux
