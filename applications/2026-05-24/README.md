# Applications — 2026-05-24 (Sunday prep for Monday send)

5 tailored bundles, each leading with the best-matching of your 3 primary
projects. Every cover-letter opener echoes 3+ phrases from the JD; every
resume rotates the lead project to the top of the Projects section.

Primary projects (reference):
- **P1 — AI Safety Red-Team Evaluation Framework** (96.8% acc, 340x cost cut, Krippendorff alpha 0.81)
- **P2 — LLM Ensemble Textbook Bias Detection** (67,500 ratings, alpha 0.84, Friedman chi-sq 42.73, p<0.001)
- **P3 — Clinical Breast Cancer Classification** (99.12% acc, 100% precision, ROC-AUC 0.9987, <100ms p95)

---

## App 1 — Anthropic · Research Engineer, Frontier Red Team (RSP Evaluations)

- **Location:** SF-based, relocation supported (apply remote-first; lead lab is your top Tier-A target)
- **JD:** https://job-boards.greenhouse.io/anthropic/jobs/5067100008 (Autonomy variant; RSP Evals variant on Anthropic careers)
- **Lead project:** P1 (AI Safety Red-Team)
- **Supporting:** P2 (Bayesian rigor), P3 (production ML)
- **Role family:** AI Safety / Red-Team / Alignment
- **Resume version:** `resume_v3_safety.pdf`

**JD phrases to echo verbatim:** "Frontier Red Team", "Responsible Scaling Policy", "catastrophic risks", "gold standard evaluations", "model organisms"

**Cover letter opener (paste, then add the next paragraph from APPLICATION_SNIPPETS.md §1 — AI Safety lab):**

> The reason I'm writing to Anthropic specifically — and not just to a generic
> frontier-lab posting — is that the Frontier Red Team's mandate to build
> *gold standard evaluations* for catastrophic risks under the Responsible
> Scaling Policy is exactly the work I've been doing on my own. In January
> 2026 I published an AI Safety Red-Team Evaluation framework that
> ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges, trains a
> 47-feature stacking classifier on 12,500 response pairs across 6 harm
> categories, and reaches 96.8% accuracy / ROC-AUC 0.9923 while cutting
> per-sample eval cost 340x (to $0.018). Krippendorff's alpha across raters
> held at 0.81, and a PyMC Bayesian hierarchy produced 95% HDI risk
> intervals per judge — the kind of uncertainty quantification an RSP
> evaluation needs to defend a "we tested this" claim.

**Resume bullets to surface (top of Projects section, in this order):**
1. P1 bullets — "AI Safety / Red-Team / Alignment" set (all 4 bullets)
2. P2 bullets — "LLM Eval / Research Engineer" set (top 2)
3. P3 bullets — "Generalist MLE / Research Engineer" set (top 1)

**Application form gotchas:**
- "Why Anthropic?" → 2-sentence answer naming Frontier Red Team + RSP Evals specifically; mention Constitutional AI as influence on your rubric design
- Visa: answer; salary: "open, targeting market for FRT RE band"
- Submit Krippendorff alpha + ROC-AUC numbers in the "what's a result you're proud of?" box

---

## App 2 — Scale AI · Strategic Projects Lead, Red Team (NYC)

- **Location:** New York, NY (on-site/hybrid; also SF/Seattle posted)
- **JD:** https://scale.com/careers/4545710005
- **Lead project:** P1 (AI Safety Red-Team)
- **Supporting:** P2 (multi-LLM scale, inter-rater reliability), P3 (rigorous benchmarking)
- **Role family:** AI Safety / Eval-Native
- **Resume version:** `resume_v3_safety.pdf`

**JD phrases to echo verbatim:** "red team", "adversarial machine learning", "frontier AI", "model testing", "actionable insights for leading AI labs"

**Cover letter opener:**

> Scale's red team sits at the seam I care about most: rigorous adversarial
> testing of frontier models that turns into actionable insights for the
> labs themselves. The clearest signal I can send for that work is the
> AI Safety Red-Team Evaluation framework I built and published — a
> 3-model judge ensemble (GPT-4o, Claude-3.5, Llama-3.2) that grades 12,500
> response pairs across 6 harm categories at 96.8% accuracy and 850
> samples/hr, with Krippendorff's alpha = 0.81 and a 340x cost reduction
> versus human annotation. It's the same shape of pipeline Scale's
> Strategic Projects team ships to its customers, with the audit-trail
> rigor (IEEE 2830-2025, SHAP per-prediction, MLflow lineage) that lets a
> lab actually act on the findings.

**Resume bullets:**
1. P1 "AI Safety / Red-Team" (all 4)
2. P2 "Trust & Safety / Content Policy" (top 2)
3. P3 "Generalist MLE" (top 1)

**Form gotchas:**
- Scale asks for a 1-paragraph "biggest adversarial-ML insight" — use the per-model blind-spot finding from P1's Bayesian hierarchy
- NYC office preference; flag willingness to be on-site 3 days/wk

---

## App 3 — Two Sigma · Quantitative Researcher: Machine Learning (NYC)

- **Location:** New York, NY (on-site)
- **JD:** https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Quantitative-Researcher-Machine-Learning/12634
- **Lead project:** P2 (LLM Bias Detection — Bayesian hierarchical model is the headline)
- **Supporting:** P1 (eval rigor, scale), P3 (model benchmarking discipline)
- **Role family:** Data Scientist (Bayesian / Causal) → Quant Research
- **Resume version:** `resume_v3_stats.pdf`

**JD phrases to echo verbatim:** "statistical techniques", "rigorous scientific approach", "hypothesis testing", "applied machine learning", "published work"

**Cover letter opener (use APPLICATION_SNIPPETS.md §2 — Data/Stats-heavy paragraph, prefixed with):**

> Two Sigma is the only place that combines the depth of statistical
> technique I'm chasing with the data scale to make those techniques bite,
> which is why I'm writing. My published work this year — an LLM-ensemble
> bias-detection study across 4,500 textbook passages and 2.5M tokens —
> is built on the habits a QR role would demand: a PyMC partial-pooling
> hierarchical model with R-hat < 1.01 and ESS > 1000, 95% HDI credible
> intervals per publisher, a Friedman omnibus (chi-squared = 42.73, p <
> 0.001) plus Nemenyi post-hoc to localize effects, and Krippendorff's
> alpha = 0.84 across the rater ensemble holding the rubric together.
> The headline finding — 3 of 5 publishers show statistically significant
> directional bias — only survives because every defensive layer is in
> place. That's the discipline I'd bring to QR work.

**Resume bullets:**
1. P2 "Data Scientist (Bayesian / Causal)" (all 3)
2. P1 "LLM Evaluation / Eval Infra" (top 2 — emphasize XGBoost meta-classifier + Bayesian hierarchy)
3. P3 "Generalist MLE" (top 2 — emphasize nested CV, 8-algorithm benchmark)

**Form gotchas:**
- Two Sigma weighs the "describe a research project" essay heavily; use 250 words on P2's prior-sensitivity analysis specifically
- They like C++/Python; emphasize Python in skills, mention C++ coursework if any

---

## App 4 — Memorial Sloan Kettering · ML Engineer, Computational Oncology (NYC)

- **Location:** New York, NY (on-site, 1275 York Ave)
- **JD:** https://careers.mskcc.org/career-areas/data-science-engineering/ (Computational Oncology programme under Sohrab Shah)
- **Lead project:** P3 (Breast Cancer Classification)
- **Supporting:** P1 (audit-grade rigor), P2 (Bayesian reasoning under uncertainty)
- **Role family:** Healthcare / Clinical ML
- **Resume version:** `resume_v3_healthcare.pdf`

**JD phrases to echo verbatim:** "computational oncology", "cancer biology", "clinical care", "mathematical modeling", "interactive visualization"

**Cover letter opener:**

> The reason I'm writing to MSK specifically is that the Computational
> Oncology programme is exactly where I want my next training cycles to
> live: clinical-care problems with computer-science depth and a culture
> that takes mathematical modeling seriously. The work I'd point to is a
> clinical-grade breast-cancer classifier I shipped earlier this year —
> 99.12% accuracy, 100% precision (zero false positives), 98.59% recall,
> ROC-AUC 0.9987 across an 8-algorithm benchmark — comfortably above the
> 90-95% range typically cited for expert reads. Just as important for
> deployment in an MSK-grade setting: SHAP explanations per prediction,
> VIF-pruned features, SMOTE-balanced training, and a FastAPI service at
> sub-100ms p95, all aligned with IEEE 2830-2025 transparency standards.
> I'd love to apply the same rigor to the computational-oncology pipelines
> coming out of Dr. Shah's group.

**Resume bullets:**
1. P3 "Healthcare / Clinical ML" (all 3) — top of section
2. P3 "Applied ML / MLE" (top 2 — adds the algorithm-benchmark + FastAPI signal)
3. P1 "ML Research Engineer / Applied Research" (top 2 — adds the Bayesian/rigour signal)

**Form gotchas:**
- MSK asks about HIPAA / PHI familiarity; answer honestly + mention IEEE 2830-2025 audit pipeline as evidence of safe-data instincts
- Visa + on-site requirement: confirm NYC availability

---

## App 5 — Hugging Face · Open-Source ML Engineer (International Remote, NYC base)

- **Location:** Remote (NYC-eligible)
- **JD:** https://apply.workable.com/huggingface/ (Open-Source ML Engineer — Transformers / Eval ecosystem)
- **Lead project:** P1 (AI Safety Red-Team — fits HF's evaluate / lm-evaluation-harness work)
- **Supporting:** P2 (multi-LLM ensemble at production scale), P3 (benchmarking discipline)
- **Role family:** LLM Evaluation / Eval Infra
- **Resume version:** `resume_v3_safety.pdf`

**JD phrases to echo verbatim:** "open-source", "Transformers", "community", "evaluation", "reproducible"

**Cover letter opener (use APPLICATION_SNIPPETS.md §1 — Eval platform variant, prefixed with):**

> The reason Hugging Face is the only "remote MLE" application I'm sending
> this week is that your open-source evaluation ecosystem — `evaluate`,
> `lm-evaluation-harness`, the Open LLM Leaderboard — is the public utility
> I've been building privately. My most recent project is a 3-model LLM
> eval harness (GPT-4o, Claude-3.5, Llama-3.2) that auto-grades 12,500
> response pairs at 96.8% accuracy and 850 samples/hr, with circuit
> breakers, async batching, MLflow tracking, and a 47-feature stacking
> meta-classifier that reconciles judge disagreement — released as a
> reproducible technical report alongside the code. That's the kind of
> contribution I'd love to land inside the HF eval stack instead of next
> to it.

**Resume bullets:**
1. P1 "LLM Evaluation / Eval Infra" (all 3) — top of section
2. P2 "LLM Eval / Research Engineer" (all 3)
3. P3 "Generalist MLE" (top 1 — adds shipping-discipline signal)

**Form gotchas:**
- HF cares about real OSS contributions; if you have HF Hub uploads, link them in the "links" field (you're authenticated as `dl1413` — confirm public repos are visible)
- "Why open source?" → tie to the reproducible-tech-report habit from all 3 projects

---

## Pre-submit checklist (run before each app)

- [ ] Resume bullet order rearranged so lead project sits at top of Projects section
- [ ] Cover letter opens with a metric hook (340x / 99.12% / Krippendorff 0.84)
- [ ] At least 3 JD phrases appear verbatim in resume + cover letter
- [ ] LinkedIn, GitHub (dl1413), Portfolio URLs live and current
- [ ] If JD lists PyTorch / JAX / Ray / vLLM, it appears in skills
- [ ] Salary expectation: "open, targeting market for the role/location"
- [ ] Work-auth answered
- [ ] JD saved as PDF in `applications/2026-05-24/jd_<n>_<company>.pdf`
- [ ] Row appended to `application_tracker.csv`

## End-of-batch action

After submitting all 5, append today's outcomes to the Weekly Review notes
(Friday 30-min block). Track callbacks by **role family** so you can kill
families with <2% response rate after 50 apps.
