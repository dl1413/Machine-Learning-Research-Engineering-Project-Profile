# Anthropic — Research Engineer, Model Evaluations

- **Location:** Remote (US)
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting projects:** LLM Ensemble Bias Detection, Breast Cancer Classification
- **JD:** https://job-boards.greenhouse.io/anthropic/jobs/4770581008
- **Portfolio:** https://github.com/dl1413/machine-learning-research-engineering-project-profile

---

## Cover Letter

Dear Anthropic Model Evaluations Team,

I'm applying for the Research Engineer, Model Evaluations role because the
work I've been doing for the last year is a smaller version of what your
team does at scale: partnering on what to measure, running the eval,
interpreting the result, and shipping the pipeline that turns those
answers into a repeatable process.

My AI Safety Red-Team Evaluation framework runs a dual-stage LLM ensemble
(GPT-4o, Claude-3.5, Llama-3.2) that annotates 12,500 response pairs
across 6 harm categories, then hands off to a stacking classifier that
hits **96.8% accuracy, 97.2% precision, and 96.1% recall** on held-out
data. The interesting number is not the accuracy — it's the reliability:
**Krippendorff's α = 0.81** across the three graders means the pipeline
holds up as a judge on its own, at **$0.018/sample vs. $6.12 for human
annotation (340× cheaper)** and **850 samples/hour**. I built the
Bayesian hierarchical layer on top (PyMC, MCMC R-hat < 1.01, 95% HDI) so
per-model risk gets uncertainty-quantified instead of point-estimated.

Two adjacent projects give me the rest of the range Anthropic evals
teams tend to need:

- **LLM Bias Detection** — 67,500 ratings across 4,500 passages, 5
  publishers, Friedman χ² = 42.73 (p < 0.001), α = 0.84. Same
  ensemble → judge → hierarchical-Bayesian pattern, applied to a
  measurement problem where the ground truth is contested rather than
  binary. That's the shape most eval work actually takes.
- **Breast Cancer Classification** — clinical-grade ML with Platt
  calibration (ECE 0.0089) and threshold policies (100% sensitivity at
  0.31). If eval outputs feed a training or deployment gate, calibration
  matters as much as accuracy — that's the muscle this project builds.

I ship the plumbing too: 80K+ API calls with circuit breakers,
exponential backoff, MLflow tracking, and IEEE 2830-2025 /
ISO/IEC 23894:2025 / EU AI Act-aligned model cards. If it were up to me
I'd start by mapping your current eval surface against those three
patterns and picking one metric where a hierarchical treatment would
sharpen the read.

Happy to walk through any of the reports; the full portfolio is linked
above.

Best,
Derek Lankeaux
MS Applied Statistics, Rochester Institute of Technology (2026)
LinkedIn: https://linkedin.com/in/derek-lankeaux

---

## One-Page Pitch (paste into "Why Anthropic?" free-text)

Anthropic's eval work is where the interesting version of statistics is
happening right now — the same measurement-under-uncertainty problems
that got me into Applied Statistics, but with a decision stakes profile
most academic work never touches. My three projects are all variants of
the same pipeline: multi-model LLM ensemble → judged with reliability
diagnostics → hierarchical Bayesian layer → SHAP / calibration on top.
I've run it at 340× cost reduction against human baselines while
holding α ≥ 0.81, on 80K+ API calls with the boring production hygiene
(circuit breakers, backoff, MLflow, IEEE / ISO / EU-AI-Act artifacts)
that lets a result survive audit. What draws me here specifically is
that your team measures capabilities against live training checkpoints —
which means the eval has to be both fast and defensible, and neither
side is negotiable. That's the tradeoff I want to be working on.

---

## Follow-up plan

- **T+0:** Submit via greenhouse; save confirmation.
- **T+3 days:** Cold email a Model Evaluations team member found on
  publication pages (arXiv, safety papers). One-paragraph note
  referencing the α = 0.81 result and asking one specific question
  about their public methodology.
- **T+10 days:** LinkedIn note if no response, then move on.
