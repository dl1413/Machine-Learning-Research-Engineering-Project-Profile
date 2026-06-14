# Data Science Proposal — Master Template & Variants

**Owner:** Derek Lankeaux, MS (Applied Statistics)
**Purpose:** Reusable, honestly-formatted proposal for winning data science
roles — direct outreach to hiring managers, scoped-role applications, and
"optional cover letter" / approach-document boxes.
**Companion docs:** `Resume_Derek_Lankeaux.md`, `APPLICATION_SNIPPETS.md`,
`JOB_APPLICATIONS_PLAYBOOK.md`

> **Ground rule:** every metric in this document is traceable to one of the
> three published technical reports in this repository. Do not add a number
> you cannot point to in a report. A proposal wins on specificity and
> credibility, not inflation — when you overstate, the first technical
> screen exposes it. Fill the `[BRACKETED]` fields per opportunity and delete
> the variants you aren't using before sending.

---

## How to use this document

1. Pick the format that matches the channel:
   - **Long-form proposal** → direct outreach to a hiring manager, or a posting
     that asks for an approach/scope document.
   - **Short-form proposal** → "optional cover letter" box or a short intro
     message (250–350 words).
2. Pick the **role-tailored variant** (Safety/Eval, Bayesian/Causal DS, or
   Applied ML/Healthcare) whose lead project best matches the JD. See the
   matrix in `JOB_APPLICATIONS_PLAYBOOK.md §1`.
3. Replace every `[BRACKET]`. Echo 3+ exact phrases from the posting.
4. Run the **Pre-send checklist** at the bottom.

---

## Part A — Long-Form Proposal (direct outreach / scoped role)

### Subject / Title
**Data Science Proposal: [SPECIFIC OUTCOME] for [COMPANY/TEAM]**

### 1. Opening — lead with the match (2–3 sentences)
Dear [HIRING MANAGER / TEAM],

I'm writing about the [ROLE TITLE] role / your need for [PROBLEM STATED IN
POSTING]. The most relevant thing I've shipped is [LEAD PROJECT — one line,
one metric]. I'd like to bring the same combination of statistical rigor and
production delivery to [COMPANY]'s [SPECIFIC TEAM/PRODUCT].

### 2. What I understand you need (restate the problem)
From the posting and [my research into COMPANY], the core problem looks like:
- [Need #1 — phrased in their words]
- [Need #2]
- [Constraint — e.g., regulated data, latency budget, small labeled set]

If I've misframed any of this, I'd welcome a 20-minute call to correct it
before going further. *(Showing you can be wrong is a credibility signal.)*

### 3. How I'd approach it (method, not hand-waving)
A phased approach I'd propose:

1. **Frame & baseline.** Define the success metric and a defensible baseline
   before modeling. Establish how we'll know the result is real (held-out
   evaluation, statistical test, calibration target).
2. **Build & evaluate.** [Modeling approach for this problem — e.g., ensemble
   benchmark, LLM-as-judge harness, hierarchical Bayesian model].
3. **Quantify uncertainty.** Report intervals, not just point estimates
   (bootstrap CIs / 95% HDI), and correct for multiple comparisons where the
   analysis demands it.
4. **Ship & monitor.** [Deployment surface — FastAPI service, eval harness,
   notebook + model card], with experiment tracking and drift monitoring.
5. **Communicate.** A stakeholder-readable readout: what we found, how
   confident we are, and what decision it supports.

### 4. Evidence I can do this (proof, drawn from published work)
Three independent, published projects back the approach above:

| Project | What it demonstrates | Headline result (from report) |
|---|---|---|
| AI Safety Red-Team Evaluation | LLM eval at scale + cost engineering | 96.8% accuracy, ROC-AUC 0.9923; $0.018/sample (340× cheaper than human annotation); Krippendorff's α = 0.81 over 12,500 pairs |
| LLM Ensemble Bias Detection | Defensible statistical inference | Bayesian hierarchical model (R-hat < 1.01); Friedman χ² = 42.73, p < 0.001; α = 0.84 over 67,500 ratings |
| Breast Cancer Classification | High-stakes predictive modeling + deployment | 99.12% accuracy, 100% precision, ROC-AUC 0.9987; FastAPI service <100ms p95 |

Full technical reports and code are in my GitHub (`dl1413`); happy to walk
through any of them.

### 5. Next step (one clear ask)
Could we schedule a short call this week? I'll come with [one concrete thing —
a sketch of the eval plan / questions about your data]. Either way, thank you
for reading.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Part B — Short-Form Proposal (platform bid / optional cover letter, 250–350 words)

Hi [NAME / TEAM],

You're looking for [PROBLEM IN THEIR WORDS]. I'm a data scientist (MS, Applied
Statistics) who ships the full workflow — framing the question, building the
model, quantifying uncertainty, and deploying the result.

The closest thing I've done to your problem: [LEAD PROJECT in one sentence with
ONE metric — e.g., *"I built a 3-model LLM ensemble that auto-grades harmful
outputs at 96.8% accuracy and 340× lower cost than human annotation, holding
inter-rater reliability at Krippendorff's α = 0.81 across 12,500 pairs."*]

For your project I'd [2–3 sentences of concrete approach tailored to the
posting — phase 1 baseline, phase 2 model/eval, how you'd prove it worked].

Why me specifically:
- **Rigor:** I report intervals, not just point estimates (bootstrap CIs / 95%
  HDI), and I test before I claim (e.g., Friedman χ² = 42.73, p < 0.001 on a
  published bias study).
- **Delivery:** my models ship — FastAPI services under 100ms p95, MLflow
  tracking, model cards, SHAP explanations.
- **Communication:** three published technical reports written for both
  reviewers and non-technical stakeholders.

If it's useful, I'm happy to walk through a [concrete first step — e.g., a
sketch of the evaluation plan, or how I'd baseline this] on a short call, so
you can judge how I think before committing to a full process. Code and reports
are on my GitHub (`dl1413`).

Happy to hop on a quick call. Thanks for your time.

— Derek Lankeaux

---

## Part C — Role-Tailored Variants (swap into Part A §1 + §4 lead row)

### C1 · AI Safety / LLM Evaluation / Red-Team
**Opening:** My most relevant work is an independent AI Safety Red-Team
Evaluation framework (published Jan 2026). It ensembles GPT-4o, Claude-3.5, and
Llama-3.2 as red-team judges and trains a stacking classifier on 47
harm-signal features — 96.8% accuracy, ROC-AUC 0.9923 over a 12,500-pair
benchmark across 6 harm categories — at 850 samples/hr and $0.018/sample (340×
cheaper than human annotation), holding Krippendorff's α = 0.81.
**Lead proof row:** AI Safety Red-Team Evaluation.
**Echo their vocabulary:** red-teaming, jailbreak detection, harm
classification, LLM-as-judge, eval harness, inter-rater reliability.

### C2 · Data Scientist (Bayesian / Causal / Experimentation)
**Opening:** The work closest to your inference needs is an LLM-ensemble bias
study: 4,500 passages, 2.5M tokens, 67,500 ratings. The headline — 3 of 5
publishers showed statistically significant directional bias (Friedman χ² =
42.73, p < 0.001) — holds because it's defended: α = 0.84 across raters and a
PyMC partial-pooling hierarchical model (R-hat < 1.01) giving 95% HDI credible
intervals per publisher.
**Lead proof row:** LLM Ensemble Bias Detection.
**Echo their vocabulary:** experimentation, A/B testing, hierarchical
modeling, MCMC, multiple-testing correction, causal inference, credible
intervals.

### C3 · Applied ML / MLE / Healthcare ML
**Opening:** A recent project that captures how I work: a clinical-grade
classifier benchmarking 8 algorithms (Random Forest, XGBoost, LightGBM,
AdaBoost, Stacking, Voting, +2) under cross-validation, landing at 99.12%
accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987 — then productionized
behind a FastAPI service at <100ms p95 with SHAP explanations and an MLflow
registry.
**Lead proof row:** Breast Cancer Classification.
**Echo their vocabulary:** ensemble methods, feature engineering, calibration,
production ML pipeline, MLOps, low-latency inference, explainability.

---

## Part D — Pre-Send Checklist

- [ ] Every `[BRACKET]` is replaced; unused variants/sections deleted
- [ ] Lead project matches the posting's primary need
- [ ] At least 3 phrases from the posting appear verbatim
- [ ] Exactly one metric in the opener; the rest live in the proof table
- [ ] Every number is traceable to a report in this repo (no rounding up, no new claims)
- [ ] One clear, low-friction next step (a call, or a concrete first step)
- [ ] LinkedIn / GitHub / Portfolio links are live
- [ ] Work authorization + location/timeline answered if the posting asks (see resume footer)
- [ ] Tone check: confident, specific, and honest — no superlatives you can't defend

---

## Anti-patterns (why proposals lose)

- **Vague approach.** "I'll use ML and AI" reads as no plan. Name the baseline,
  the method, and the test.
- **All claim, no proof.** Pair every capability statement with a traceable
  result or a code link.
- **No clear ask.** End with one concrete next step, not "let me know."
- **Inflated numbers.** A single unverifiable stat poisons the whole proposal
  at the technical screen. Specific and true beats big and shaky.
</content>
</invoke>
