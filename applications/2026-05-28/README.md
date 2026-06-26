# Daily Application Batch — 2026-05-28 (5 roles, NYC + Remote)

**Owner:** Derek Lankeaux · **Target:** 5 tailored applications/day, NYC on-site/hybrid or US-remote.

Each package below leads with the project that best matches the role and uses the
other two as supporting evidence. Cover letters are paste-ready. Resume guidance
points to the exact bullet block in `../../APPLICATION_SNIPPETS.md`.

> **Before you hit submit (playbook rule):** open the live JD and (1) confirm the
> posting is still open, (2) echo **3+ exact phrases** from that JD verbatim in
> your resume + cover letter, (3) confirm the location/remote policy, (4) answer
> work-auth + salary. The keyword lists below come from public listing summaries,
> not the full JD — verify against the real posting.

---

## 1 — 10a Labs · AI Red Teamer (Entry Level) · **Remote**

- **Apply:** https://job-boards.greenhouse.io/10alabs/jobs/4002004009
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework
- **Supporting:** P2 (LLM bias detection), P3 (rigor)
- **Resume:** put `APPLICATION_SNIPPETS.md → Project 1 → "AI Safety / Red-Team / Alignment"` bullets at top of Projects; keep P2 "LLM Eval / Research Engineer" bullets second.
- **Keywords to mirror (verify in JD):** AI red teaming, LLM, harm categories, Python, evaluation, prompt, vulnerability, content policy.

**Cover letter (paste-ready):**

> Dear 10a Labs Hiring Team,
>
> I'm applying for your AI Red Teamer role. My most relevant work is an
> independent AI Safety Red-Team Evaluation framework I built and published in
> January 2026: it ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges
> and trains a stacking classifier on 47 harm-signal features, reaching 96.8%
> accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm
> categories. The pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost
> reduction versus human annotation — while holding inter-rater reliability at
> Krippendorff's alpha = 0.81.
>
> I pair red-team intuition with measurement discipline: a companion LLM-ensemble
> study ran 67,500 bias ratings over 4,500 passages and surfaced statistically
> significant publisher bias (Friedman chi-squared = 42.73, p < 0.001), and a
> clinical-grade classifier I shipped hit 99.12% accuracy with zero false
> positives. I'm comfortable in Python end-to-end, from prompt design to the
> async API layer (circuit breakers, exponential backoff) that kept those
> pipelines running unattended.
>
> I'm based in/available for NYC and open to remote, targeting a 2026 start as I
> wrap my MS in Applied Statistics at RIT. Code and all three technical reports
> are on my GitHub (dl1413). The red-team eval is the fastest way to see how I
> think about adversarial evaluation.
>
> Best,
> Derek Lankeaux

---

## 2 — Mount Sinai (Windreich Dept. of AI & Human Health) · ML Engineer III, Generative AI · **New York, NY**

- **Apply:** https://careers.mountsinai.org/jobs/3035197
- **Lead project:** P3 — Clinical-Grade Breast Cancer ML Classification System
- **Supporting:** P1 (GenAI eval rigor), P2 (Bayesian uncertainty)
- **Resume:** put `Project 3 → "Healthcare / Clinical ML"` bullets at top; follow with `Project 1 → "LLM Evaluation / Eval Infra"` to cover the GenAI angle.
- **Keywords to mirror (verify in JD):** generative AI, clinical, healthcare, machine learning pipeline, deployment, explainability, electronic health records.

**Cover letter (paste-ready):**

> Dear Mount Sinai Hiring Team,
>
> I'm excited to apply for the Machine Learning Engineer III, Generative AI role
> in the Windreich Department of AI & Human Health. The work most relevant to your
> team is a clinical-grade breast-cancer classifier I shipped this year: I
> benchmarked 8 algorithms end-to-end — Random Forest through stacking ensembles —
> and landed at 99.12% accuracy with 100% precision (zero false positives), 98.59%
> recall, and ROC-AUC 0.9987, above the 90–95% range typically cited for human
> expert reads. For clinical deployment it ships with SHAP explanations per
> prediction, VIF-pruned features, and a FastAPI service under 100ms p95, aligned
> with IEEE 2830-2025 transparency standards.
>
> On the generative-AI side, I built and published an LLM evaluation framework
> that auto-grades 12,500 response pairs across GPT-4o, Claude-3.5, and Llama-3.2
> at 96.8% accuracy, with a PyMC Bayesian hierarchical model producing 95% HDI
> risk intervals — exactly the uncertainty quantification clinical decision
> support needs. I own the full workflow: framing the question, building the
> model, quantifying uncertainty, and communicating to non-technical
> stakeholders.
>
> I'm NYC-based and finishing my MS in Applied Statistics at RIT (2026). Reports
> and code are on my GitHub (dl1413) — I'd welcome the chance to bring this rigor
> to human-health AI at Mount Sinai.
>
> Best,
> Derek Lankeaux

---

## 3 — SmarterDx · Senior Data Analyst, Data Science & AI Research · **Remote (US)**

- **Apply:** https://job-boards.greenhouse.io/smarterdx/jobs/5053199007
- **Lead project:** P2 — LLM Ensemble Bias Detection (Bayesian rigor)
- **Supporting:** P3 (clinical ML), P1 (LLM eval)
- **Resume:** put `Project 2 → "Data Scientist (Bayesian / Causal)"` bullets at top; follow with `Project 3 → "Healthcare / Clinical ML"`.
- **Keywords to mirror (verify in JD):** Bayesian statistics, healthcare, clinical data, SQL, experimentation, statistical modeling, LLM.

**Cover letter (paste-ready):**

> Dear SmarterDx Team,
>
> I'm applying for the Senior Data Analyst, Data Science & AI Research role.
> SmarterDx sits exactly where my strengths meet — clinical data plus defensible
> statistics. One project I'd point to is an LLM-ensemble study I ran last
> quarter: 4,500 passages, 2.5M tokens, 67,500 ratings from GPT-4o / Claude-3.5 /
> Llama-3.2. The headline finding — significant directional bias in 3 of 5 sources
> (Friedman chi-squared = 42.73, p < 0.001) — only holds because the pipeline was
> built to defend it: Krippendorff's alpha = 0.84 across raters, and a PyMC
> partial-pooling hierarchical model with R-hat < 1.01 producing 95% HDI credible
> intervals rather than point estimates.
>
> That Bayesian-first habit carries into clinical modeling: my breast-cancer
> classifier reached 99.12% accuracy and 100% precision under nested
> cross-validation with SHAP explanations and a fairness audit. I'm fluent in SQL
> and own the full analysis loop — framing the question, writing the query,
> modeling, quantifying uncertainty, and translating it for stakeholders.
>
> I'm available remote, finishing my MS in Applied Statistics at RIT (2026). Code
> and three technical reports are on my GitHub (dl1413).
>
> Best,
> Derek Lankeaux

---

## 4 — Blue Rose Research · ML / Data Scientist (Statistical Modeling & AI Systems) · **New York, NY**

- **Apply:** https://job-boards.greenhouse.io/blueroseresearch/jobs/5688383004
- **Lead project:** P2 — LLM Ensemble Bias Detection (hierarchical modeling)
- **Supporting:** P1 (multi-model AI systems), P3 (modeling rigor)
- **Resume:** put `Project 2 → "Data Scientist (Bayesian / Causal)"` bullets at top; follow with `Project 1 → "ML Research Engineer / Applied Research"`.
- **Keywords to mirror (verify in JD):** statistical modeling, Bayesian, hierarchical models, AI systems, Python, inference, partial pooling.

**Cover letter (paste-ready):**

> Dear Blue Rose Research Team,
>
> I'm applying for your ML / Data Scientist (Statistical Modeling & AI Systems)
> role. The work that best maps to it is an LLM-ensemble modeling study where the
> statistics did the heavy lifting: across 4,500 passages and 67,500 ratings, I
> fit a PyMC Bayesian hierarchical model with partial pooling that reached MCMC
> convergence (R-hat < 1.01, ESS > 1000) and produced 95% HDI credible intervals
> per group, backed by a Friedman omnibus test (chi-squared = 42.73, p < 0.001)
> and post-hoc Nemenyi comparisons to localize effects.
>
> I also build the AI systems that feed those models: a 3-LLM ensemble (GPT-4o,
> Claude-3.5, Llama-3.2) auto-grading 12,500 pairs at 96.8% accuracy, stacked into
> an XGBoost meta-classifier over 47 engineered features with full MLflow lineage.
> I pick models by evidence, quantify uncertainty honestly, and ship reproducible
> pipelines — the combination Blue Rose's modeling work seems to demand.
>
> I'm NYC-based, finishing my MS in Applied Statistics at RIT (2026). Code and
> reports are on my GitHub (dl1413).
>
> Best,
> Derek Lankeaux

---

## 5 — Innodata · AI/ML Research Engineer (LLM Training & Evaluation) · **Remote**

- **Find/apply:** Innodata Careers — https://innodata.com/careers/ (search "AI/ML Research Engineer" / "LLM Training & Evaluation"); also listed via Glassdoor. **Confirm the live req before applying.**
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework (eval infra)
- **Supporting:** P2 (multi-LLM eval at scale), P3 (model selection rigor)
- **Resume:** put `Project 1 → "LLM Evaluation / Eval Infra"` bullets at top; follow with `Project 2 → "LLM Eval / Research Engineer"`.
- **Keywords to mirror (verify in JD):** LLM training, evaluation pipeline, fine-tuning, human-in-the-loop, reproducibility, metrics quality, foundation models.

**Cover letter (paste-ready):**

> Dear Innodata Hiring Team,
>
> I'm applying for the AI/ML Research Engineer (LLM Training & Evaluation) role.
> I've spent the last few months building exactly this kind of evaluation
> infrastructure. My AI Safety Red-Team framework is a production LLM eval harness
> — GPT-4o, Claude-3.5, Llama-3.2 — that auto-grades 12,500 response pairs at
> 96.8% accuracy and 850 samples/hr, with circuit breakers, async batching, and
> MLflow run tracking baked in. Cost per sample landed at $0.018, a 340x reduction
> versus human review. The interesting layer is a 47-feature stacking
> meta-classifier that reconciles disagreement between the three judges and
> surfaces per-model blind spots via Bayesian hierarchical modeling.
>
> I care about the practical realities your team lives in — reproducibility,
> debugging, metrics quality, iteration speed. A second multi-LLM study pushed
> 67,500 ratings over 2.5M tokens with full lineage and inter-rater reliability of
> Krippendorff's alpha = 0.84, and I validate human-in-the-loop rubrics rather
> than trusting raw model output.
>
> I'm available remote, finishing my MS in Applied Statistics at RIT (2026). Code
> and three technical reports are on my GitHub (dl1413).
>
> Best,
> Derek Lankeaux

---

## After submitting

For each role you actually submit: change `status` in `../../application_tracker.csv`
from `ready_to_submit` to `submitted` and set the date. Save the JD as a PDF in
this folder for interview prep. On Friday, run the weekly review in
`../../JOB_APPLICATIONS_PLAYBOOK.md §6`.

**Tomorrow:** ask me "build today's batch of 5" and I'll pull 5 fresh roles and
generate the next set.
