# Cover Letter - Hugging Face, Cloud Machine Learning Engineer

**Location:** US Remote
**Lead project:** AI Safety Red-Team (eval infra) + LLM Bias Detection (production scale)
**JD link:** https://apply.workable.com/huggingface/j/A3879724CD

---

Dear Hugging Face team,

I'd like to be considered for the Cloud Machine Learning Engineer role
because Hugging Face is where I already do a lot of my day-to-day work:
Transformers, Datasets, Accelerate, and the Hub are threaded through my
last two research projects. This letter walks through the piece of work
most relevant to the role - a production LLM-ensemble eval pipeline -
and what I'd want to bring into your cloud/eval stack.

I recently shipped a 3-model LLM eval harness (GPT-4o, Claude-3.5,
Llama-3.2 via Hugging Face) that auto-grades 12,500 AI response pairs
at 96.8% accuracy and 850 samples/hr. Cost per sample landed at $0.018,
a 340x reduction versus human review, while inter-rater reliability held
at Krippendorff's alpha = 0.81. The interesting engineering was the
production plumbing: async API integration, circuit breakers,
exponential backoff, MLflow run tracking, and reproducible artifacts -
the scaffolding cloud ML customers keep asking for.

A second project extends the same pattern to bias detection: 4,500
textbook passages, 2.5M tokens, 67,500 LLM ratings, all rolled up
through a PyMC Bayesian hierarchical model (R-hat < 1.01) that surfaces
publisher-level bias at Friedman chi-squared = 42.73, p < 0.001. Both
pipelines ship with model cards and IEEE 2830-2025-aligned audit
trails - the responsible-AI defaults Hugging Face has been pushing the
industry toward.

What that translates to for a Cloud MLE role:

- **Real production hygiene** for LLM workloads: batching, retries,
  cost accounting, MLflow lineage, model registry.
- **A working knowledge of open-source ML.** Transformers, Datasets,
  Accelerate, PEFT, PyTorch 2 - I'm a heavy user and would happily
  contribute upstream.
- **Statistical + eval literacy.** MS in Applied Statistics at RIT (2026),
  plus two published multi-LLM evaluation projects with rigor above the
  usual "vibe check" bar.
- **A cover-letter answer to "why open-source."** I've published every
  project's methodology and code, and I think eval standards move faster
  when they're open - which is Hugging Face's whole thesis.

I'm available for US remote and targeting a 2026 start once I finish my
MS. Portfolio, code, and three technical reports are on GitHub at
dl1413.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
