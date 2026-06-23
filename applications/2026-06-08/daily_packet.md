# Daily Application Packet — 2026-06-08

**Owner:** Derek Lankeaux
**Target:** 5 NYC + remote roles, each tailored to one lead project from the
3-project profile (Red-Team Eval, Bias Detection, Breast Cancer
Classification). Mix balances all 3 projects across the day.

| # | Company | Role | Loc | Lead Project | Support |
|---|---|---|---|---|---|
| 1 | Anthropic | AI Safety Fellow (May/Jul 2026 cohort) | Remote / SF / NYC | P1 Red-Team | P2, P3 |
| 2 | Mount Sinai (Windreich Dept) | ML Engineer III — Generative AI | NYC (on-site) | P3 Clinical | P1 |
| 3 | QuantCo | Research Scientist (Stats / Bayesian) | Hybrid NYC / Remote | P2 Bayesian | P1 |
| 4 | Scale Labs | ML Research Engineer — Eval & Safety | NYC / SF / Remote | P1 Red-Team | P2 |
| 5 | Ring (Amazon) | AI/LLM Evaluation & Alignment SWE | Remote | P1 Red-Team | P2 |

---

## 1 — Anthropic, AI Safety Fellow (May/Jul 2026 cohort)

**JD:** https://www.anthropic.com/careers
**Why fit:** Fellowship is explicitly research-mentorship for engineers and
researchers (incl. MS-level). Areas listed: scalable oversight, adversarial
robustness & AI control, model organisms, AI security. Project 1 hits 3 of
the 5 listed tracks.

**Resume bullet order:** Project 1 → Project 2 → Project 3.

**Cover letter (paste-ready):**

> Dear Anthropic Fellows Program team,
>
> I'm applying to the Fellows Program because the program's adversarial-robustness
> and AI-control tracks are the exact problems I've been building toward as an
> independent researcher.
>
> My most relevant work is the AI Safety Red-Team Evaluation framework I
> published in January 2026. It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as
> red-team judges and trains a stacking classifier on 47 harm-signal features,
> reaching 96.8% accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark
> across 6 harm categories. The pipeline runs at 850 samples/hr for
> $0.018/sample — a 340x cost reduction vs human annotation — while holding
> inter-rater reliability at Krippendorff's alpha = 0.81. I paired it with a
> PyMC Bayesian hierarchical model that produces 95% HDI risk intervals per
> judge, and shipped the whole thing under IEEE 2830-2025 audit-trail
> requirements.
>
> Two supporting projects under the same rigor bar: (a) a multi-LLM
> bias-detection study (67,500 ratings, 4,500 passages, p < 0.001 across
> publishers with alpha = 0.84), and (b) a clinical-grade ensemble classifier
> (99.12% accuracy, ROC-AUC 0.9987, FastAPI <100ms p95). Three reports, three
> code releases — everything is reproducible.
>
> I'm finishing an MS in Applied Statistics at RIT in 2026, based in / available
> for NYC and open to remote. The Fellowship's four-month structure plus
> compute support is exactly the runway I want for scalable-oversight work.
> Happy to walk through the red-team eval — probably the fastest way to see
> how I think.
>
> Best,
> Derek Lankeaux

**JD keywords to echo verbatim:** adversarial robustness, AI control,
scalable oversight, mechanistic interpretability, model organisms,
red-teaming, AI safety.

---

## 2 — Mount Sinai Windreich Dept, ML Engineer III, Generative AI

**JD:** https://careers.mountsinai.org/jobs/3035197
**Why fit:** NYC on-site, GenAI in clinical setting. Project 3 (clinical
classifier, IEEE 2830-2025, SHAP) is the lead; Project 1 supplies the GenAI
credentials.

**Resume bullet order:** Project 3 → Project 1 → Project 2.

**Cover letter:**

> Dear Windreich Department Hiring Team,
>
> Two of my three independent research projects map directly to the Windreich
> Department's mandate, so I'm applying for the Machine Learning Engineer III —
> Generative AI role with concrete artifacts rather than aspirations.
>
> The headline project: a clinical-grade breast-cancer classifier I shipped
> this year. I benchmarked 8 algorithms end-to-end — Random Forest through
> stacking ensembles — and landed at 99.12% accuracy with 100% precision
> (zero false positives), 98.59% recall, and ROC-AUC 0.9987, comfortably
> above the 90–95% human-expert read range. Just as important for clinical
> deployment: the pipeline ships with SHAP explanations per prediction,
> VIF-pruned features, SMOTE class balancing, and a FastAPI service under
> 100ms p95, all aligned with IEEE 2830-2025 transparency standards.
>
> On the generative-AI side, my AI Safety Red-Team Evaluation framework
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as harm-detection judges,
> reaches 96.8% accuracy on a 12,500-sample benchmark, and runs at $0.018
> per sample — the same eval rigor you'd want on any clinical LLM pipeline.
> Inter-rater reliability stayed at Krippendorff's alpha = 0.81 and I
> quantified judge uncertainty with a PyMC Bayesian hierarchy.
>
> I'm finishing an MS in Applied Statistics at RIT in 2026 and would relocate
> for a Mount Sinai on-site role. I'd love to bring the clinical-deployment
> habits — SHAP everywhere, audit trails, FastAPI latency budgets — to
> Windreich's GenAI portfolio.
>
> Best,
> Derek Lankeaux

**JD keywords:** generative AI, clinical, HIPAA, EHR, deployment, latency,
explainability, fairness.

---

## 3 — QuantCo, Research Scientist (Statistical Inference / Bayesian)

**JD:** https://www.builtinnyc.com/job/research-scientist/4670307
**Why fit:** JD asks for Bayesian inference and probabilistic ML in
high-dimensional systems. Project 2 (PyMC hierarchical, MCMC, partial
pooling, Friedman + Nemenyi) is the cleanest match.

**Resume bullet order:** Project 2 → Project 1 → Project 3.

**Cover letter:**

> Dear QuantCo Research Team,
>
> The Research Scientist posting reads like a description of how I already
> work, so I'd like to be considered.
>
> The clearest demonstration is an LLM-ensemble bias-detection study I ran
> last quarter: 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings
> from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of
> 5 publishers showed statistically significant directional bias (Friedman
> chi-squared = 42.73, p < 0.001) — only holds because the pipeline was
> built to defend it: Krippendorff's alpha = 0.84 across raters, 92%
> pairwise correlation, and a PyMC partial-pooling hierarchical model with
> R-hat < 1.01 producing 95% HDI credible intervals per publisher. Friedman
> omnibus + post-hoc Nemenyi pairwise comparisons localized the effects.
>
> Two supporting projects keep the inference muscle visible across problem
> types: an LLM safety-eval framework (96.8% accuracy, Bayesian judge-
> uncertainty model, $0.018/sample) and a clinical-grade classifier (99.12%
> accuracy, ROC-AUC 0.9987). All three reports are public.
>
> Background: MS Applied Statistics at RIT (2026), specialization in Bayesian
> methods and experimental design. Based in / available for NYC, open to
> QuantCo's hybrid setup. Happy to walk through the priors-and-posteriors
> work whenever convenient.
>
> Best,
> Derek Lankeaux

**JD keywords:** Bayesian inference, probabilistic modeling, high-dimensional
statistics, causal inference, hierarchical, partial pooling.

---

## 4 — Scale Labs, ML Research Engineer (Eval / Safety / Benchmarking)

**JD:** https://labs.scale.com/jobs
**Why fit:** Scale Labs explicitly advances AI evaluation, safety, and
benchmarking. Project 1 is the lead; Project 2 shows multi-LLM eval at
production scale.

**Resume bullet order:** Project 1 → Project 2 → Project 3.

**Cover letter:**

> Dear Scale Labs team,
>
> I'm applying for the ML Research Engineer role because evaluation,
> benchmarking, and safety are the three things I've spent the last six
> months building — independently, end-to-end, with everything published.
>
> My AI Safety Red-Team Evaluation framework auto-grades 12,500 response
> pairs from a GPT-4o / Claude-3.5 / Llama-3.2 ensemble across 6 harm
> categories. It hits 96.8% accuracy, ROC-AUC 0.9923, and Krippendorff's
> alpha = 0.81, running at 850 samples/hr for $0.018/sample (340x cheaper
> than human annotation) with circuit breakers, exponential backoff, and
> MLflow lineage end-to-end. A stacking meta-classifier on 47 engineered
> harm-signal features reconciles inter-judge disagreement; a PyMC Bayesian
> hierarchy surfaces per-model blind spots via 95% HDI.
>
> The companion bias-detection project ran the same ensemble across 4,500
> textbook passages and 67,500 ratings, holding alpha = 0.84 and surfacing
> publisher bias at p < 0.001 — the kind of result that needs the
> reliability scaffolding to be trusted.
>
> Background: MS Applied Statistics at RIT (2026). Based in / available for
> NYC, open to remote or SF on-site. I'd love to help Scale Labs ship the
> next generation of public LLM benchmarks.
>
> Best,
> Derek Lankeaux

**JD keywords:** evaluation harness, benchmark, LLM-as-judge, red-team,
alignment, safety eval, dataset.

---

## 5 — Ring (Amazon), AI/LLM Evaluation & Alignment Software Engineer

**JD:** https://jobgether.com/offer/692fcb1511d3dca7d9e048ad-ai-llm-evaluation-alignment-software-engineer
**Why fit:** Remote, focused on LLM evaluation and alignment infrastructure
in a product setting. Project 1 lead, Project 2 support.

**Resume bullet order:** Project 1 → Project 2 → Project 3.

**Cover letter:**

> Dear Ring Hiring Team,
>
> I'm applying for the AI/LLM Evaluation & Alignment Software Engineer role
> because the eval-harness-meets-production-engineering scope is exactly the
> hybrid I've been building toward.
>
> My most relevant project is an AI Safety Red-Team Evaluation framework I
> shipped in January 2026: a 3-model LLM eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
> 850 samples/hr, with circuit breakers, async batching, exponential
> backoff, and MLflow tracking baked in. Cost per sample landed at $0.018,
> a 340x reduction versus human review. A stacking layer on 47 harm-signal
> features reconciles disagreement across the three judges and surfaces
> per-model blind spots via a PyMC Bayesian hierarchy.
>
> Companion project at the same scale: a multi-LLM textbook-bias-detection
> pipeline (67,500 ratings, 2.5M tokens, alpha = 0.84, p < 0.001 across
> publishers). And a clinical-grade classifier (99.12% accuracy, ROC-AUC
> 0.9987, FastAPI <100ms p95) showing the production-engineering side.
>
> Background: MS Applied Statistics at RIT (2026), available for remote
> with a NYC base. I'd be excited to bring the eval-harness habits — async
> batching, circuit breakers, MLflow lineage, judge-uncertainty modeling —
> to Ring's alignment work.
>
> Best,
> Derek Lankeaux

**JD keywords:** LLM evaluation, alignment, red-team, eval harness,
LLM-as-judge, prompt engineering, production ML.

---

## Submission Checklist (run before hitting "submit" on each)

- [ ] Resume reorder matches the **Lead Project** column above
- [ ] At least 3 verbatim JD keywords appear in resume + cover letter
- [ ] One-line metric hook in the opener (340x / 99.12% / alpha=0.84)
- [ ] LinkedIn, GitHub, Portfolio URLs live
- [ ] Salary expectation: "open, targeting market for role/location"
- [ ] Work authorization: "Authorized to work in the United States"
- [ ] Save JD as PDF into this same folder once submitted
- [ ] Append row to `application_tracker.csv`
