# 5. Driver AI — Applied Data Scientist, LLM Evaluation

- **Location:** Remote (US)
- **Submit:** https://www.remoterocketship.com/us/company/driver-ai/jobs/applied-data-scientist-llm-evaluation-united-states-remote/
- **Role family:** LLM Evaluation / Eval Engineering
- **Lead project:** P1 — AI Safety Red-Team Evaluation · **Supporting:** P2 (eval at scale + Bayesian), P3

## JD keywords to echo (verbatim)
LLM evaluation · LLM-as-judge · eval harness · benchmarks · metrics · inter-rater reliability · applied data scientist

## Cover letter

Dear Driver AI Team,

This role is a direct match for what I've been building. I recently shipped a
3-model **LLM evaluation** harness — GPT-4o, Claude-3.5, Llama-3.2 — that
auto-grades 12,500 response pairs at 96.8% accuracy and 850 samples/hr, with
circuit breakers, async batching, and MLflow tracking baked in. The interesting
part is the **LLM-as-judge** design: rather than trusting one judge, I stack all
three into a 47-feature meta-classifier that reconciles disagreement and surfaces
per-model blind spots, and I report **inter-rater reliability** (Krippendorff's
alpha = 0.81) so the **metrics** are defensible.

I think about **eval harness** quality as a statistics problem, not just an
engineering one. My companion bias-detection study ran 67,500 ratings across
4,500 passages and used a PyMC partial-pooling model (R-hat < 1.01) to produce
95% HDI credible intervals — turning "the model seems biased" into a defensible
**benchmark** result at p < 0.001. As an **applied data scientist** I'd bring
both halves: the throughput to run evals at scale and the rigor to make their
numbers trustworthy.

I'm open to remote (US-based), targeting a 2026 start once I wrap my MS in
Applied Statistics at RIT. Code and three technical reports are on GitHub (dl1413).

Best,
Derek Lankeaux

## Resume — Projects section (lead order for this role)

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Built LLM-as-judge eval harness over 12,500 pairs: 96.8% accuracy, ROC-AUC 0.9923, 850 samples/hr, MLflow lineage
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier on 47 features; quantified judge disagreement with a Bayesian hierarchical model (95% HDI)
- Held inter-rater reliability at Krippendorff's alpha = 0.81; $0.018/sample (340x cheaper than human eval)

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- 67,500 ratings / 2.5M tokens; PyMC partial pooling (R-hat < 1.01, ESS > 1000); significant bias at p < 0.001; 92% pairwise inter-LLM correlation

**Clinical-Grade Breast Cancer Classification** — *Independent Research, Jan 2026*
- 8-algorithm benchmark; 99.12% accuracy / 100% precision / ROC-AUC 0.9987; FastAPI < 100ms p95
