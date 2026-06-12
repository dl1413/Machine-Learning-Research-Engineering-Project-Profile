# Daily Application Packet — 2026-06-12 (Friday)

**Owner:** Derek Lankeaux
**Target:** 5 tailored applications (NYC + Remote)
**Cadence ref:** `JOB_APPLICATIONS_PLAYBOOK.md`

This packet is a queue of 5 pre-tailored applications for **you to submit**
through each company's portal. I cannot push the "Apply" button on your
behalf — every portal needs your login, resume upload, and answers to
employer-specific questions. What's below is the per-role tailoring work
(lead project, JD-matched bullets, paste-ready cover letter), so submission
takes minutes, not hours.

---

## Today's Slate

| # | Company | Role | Location | Lead Project | Fit | Tier |
|---|---|---|---|---|---|---|
| 1 | Anthropic | Research Engineer / Scientist, Alignment Science | Remote / SF / NYC | P1 — Red-Team | Strong | A |
| 2 | Anthropic | Fellows Program — AI Security | Remote-friendly | P1 — Red-Team | Excellent (designed for emerging talent) | A |
| 3 | Hugging Face | ML Engineer — LLM Evaluation | US Remote | P1 — Red-Team / Eval | Strong | B |
| 4 | Flatiron Health | Machine Learning Engineer | NYC | P3 — Breast Cancer | Strong | D |
| 5 | Headway | Senior Staff DS — Bayesian Experimentation & Causal Inference | NYC | P2 — Bias Detection | **STRETCH** (12+ yrs req'd; sub if you want to honor the playbook's "skip Staff/Principal" rule) | C |

> **Playbook note:** Pick 5 of these and submit. If you skip Headway (Staff
> level), pull a substitute from Tier A/B at the same role family — the rest
> of the packet is ready to ship.

---

## 1. Anthropic — Research Engineer / Scientist, Alignment Science

- **JD link:** https://job-boards.greenhouse.io/anthropic/jobs/4009165008
- **Location:** Remote-friendly / SF / NYC
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting:** LLM Bias Detection (Bayesian rigor)
- **JD phrases to mirror:** "alignment", "responsible scaling policy", "red-team", "model behavior", "experimental design", "reproducible research"
- **Resume reorder:** Project 1 → Project 2 → Project 3. Skills section: PyMC, MCMC, SHAP, MLflow.

### Cover letter (paste-ready)

> Dear Anthropic Alignment Science team,
>
> My most relevant work for the Alignment Science team is an independent AI
> Safety Red-Team Evaluation framework I built and published in early 2026.
> It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and
> trains a stacking classifier on 47 harm-signal features, reaching 96.8%
> accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm
> categories. The pipeline runs at 850 samples/hr for $0.018/sample — a 340x
> cost reduction versus human annotation — while holding inter-rater
> reliability at Krippendorff's alpha = 0.81. I paired it with a PyMC
> Bayesian hierarchical model that produces 95% HDI risk intervals per
> judge, and shipped the whole thing under IEEE 2830-2025 audit-trail
> requirements. I'd love to bring that combination of eval rigor and
> production throughput to alignment experiments that feed into Anthropic's
> Responsible Scaling Policy.
>
> I'm finishing my MS in Applied Statistics at RIT and am available for
> remote, NYC, or SF. Portfolio, code, and the three technical reports
> referenced above are on my GitHub (dl1413). Happy to walk through any of
> them — the red-team eval is probably the fastest way to see how I think
> about alignment.
>
> Best,
> Derek Lankeaux

### Application-form answers (common Anthropic fields)

- **Why Anthropic?** Constitutional AI and RSP are the operational
  scaffolding I tried to mirror in my own red-team framework — Anthropic is
  where that work actually shapes deployment decisions.
- **Hardest technical problem you've solved:** Reconciling disagreement
  across GPT-4o / Claude-3.5 / Llama-3.2 judges on borderline harm cases.
  Solved with a 47-feature stacking meta-classifier + PyMC hierarchical
  model that surfaced per-model blind spots via 95% HDI intervals.
- **Salary expectation:** Open, targeting market for the role/location.
- **Work authorization:** Authorized to work in the US.

---

## 2. Anthropic — Fellows Program, AI Security

- **JD link:** https://www.anthropic.com/careers/jobs/5030244008
- **Location:** Remote-friendly
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting:** LLM Bias Detection
- **Why this fits you best on paper:** Fellows is explicitly for "promising
  technical talent, regardless of previous experience" — your MS + 3
  published reports + red-team artifact map cleanly to the program's intent.
- **JD phrases to mirror:** "AI security", "mentorship", "research
  contributions", "policy-relevant", "Constitutional AI"

### Cover letter / statement of interest (paste-ready)

> The Fellows Program looks like the right next step for me because the
> work I've been doing on my own — three published technical reports in the
> last six months on red-team evaluation, LLM-ensemble bias detection, and
> clinical-grade ML — has hit the natural ceiling of "what one person can
> do without a research community around them."
>
> The artifact most relevant to AI Security is a dual-stage red-team
> evaluation framework: a 3-model LLM ensemble (GPT-4o, Claude-3.5,
> Llama-3.2) that auto-grades 12,500 response pairs across 6 harm
> categories at 96.8% accuracy (ROC-AUC 0.9923), Krippendorff's alpha =
> 0.81 inter-rater reliability, and $0.018/sample throughput — a 340x cost
> cut versus human annotation. I extended it with a PyMC Bayesian
> hierarchical model that yields 95% HDI risk intervals per judge,
> surfacing systematic blind spots per model family — exactly the kind of
> structural-disagreement analysis that matters for security work.
>
> What I want out of the Fellowship: collaborators who push the
> methodological standards higher, exposure to threat models I haven't yet
> encountered in solo work, and the chance to translate Constitutional AI
> and RSP into measurable security guarantees. I'd be ready to start
> immediately after my MS wraps in 2026.
>
> Best,
> Derek Lankeaux

---

## 3. Hugging Face — ML Engineer, LLM Evaluation (Remote)

- **JD link:** https://apply.workable.com/huggingface/ (filter: LLM
  Evaluation; if internship-only, also apply to senior eval engineer roles
  on same board)
- **Location:** US Remote
- **Lead project:** AI Safety Red-Team Evaluation (Eval-Infra angle)
- **Supporting:** LLM Bias Detection
- **JD phrases to mirror:** "Open LLM Leaderboard", "evaluation harness",
  "LLM-as-judge", "reproducible benchmarks", "community", "open-source"

### Cover letter (paste-ready)

> The reason I'm writing about the LLM Evaluation role is that I've spent
> the last six months building exactly the kind of evaluation infrastructure
> the Open LLM Leaderboard depends on — just for safety/harm signals
> instead of general capability.
>
> Concretely: a 3-model judge ensemble (GPT-4o, Claude-3.5, Llama-3.2) over
> 12,500 response pairs, scored on 6 harm categories with a 47-feature
> stacking meta-classifier. The eval harness sustains 850 samples/hr at
> $0.018/sample with circuit breakers, async batching, and MLflow run
> tracking — and the meta-classifier hits 96.8% accuracy / ROC-AUC 0.9923
> against a held-out human-labeled benchmark, with Krippendorff's alpha =
> 0.81 across the three LLM judges. A PyMC Bayesian hierarchical layer
> surfaces per-judge disagreement structure (95% HDI) so the leaderboard
> wouldn't be hiding consensus-vs-coverage trade-offs.
>
> I'd be excited to bring that to the Leaderboard's reliability scaffolding,
> add safety / harm tracks, and document the methodology in the open
> tradition Hugging Face is known for. I'm finishing my MS in Applied
> Statistics at RIT and ready for a fully remote start in 2026.
>
> Best,
> Derek Lankeaux

---

## 4. Flatiron Health — Machine Learning Engineer (NYC)

- **JD link:** https://builtin.com/job/machine-learning-engineer/4507605
  (also: https://www.builtinnyc.com/job/machine-learning-engineer/2914468 —
  apply to whichever is still active)
- **Location:** NYC (hybrid)
- **Lead project:** Breast Cancer Classification
- **Supporting:** AI Safety Red-Team (for production-rigor signal)
- **JD phrases to mirror:** "oncology", "clinical research", "patient
  outcomes", "production model", "MLOps", "explainability", "regulatory"

### Cover letter (paste-ready)

> The work most relevant to Flatiron is a clinical-grade breast-cancer
> classifier I shipped earlier this year. I benchmarked 8 algorithms
> end-to-end — Random Forest through stacking and voting ensembles — and
> landed at 99.12% accuracy with 100% precision (zero false positives),
> 98.59% recall, and ROC-AUC 0.9987, comfortably above the 90-95% range
> typically cited for human expert reads on the same problem.
>
> Just as important for clinical deployment: the pipeline ships with SHAP
> explanations per prediction, VIF-pruned features, Platt-calibrated
> probabilities (ECE 0.0089), and a FastAPI service under 100ms p95, all
> aligned with IEEE 2830-2025 transparency standards. I also adopted
> threshold-tuning policies — e.g., a 0.31 decision boundary that holds
> 100% sensitivity for screening — which is the kind of context-adaptive
> decision policy oncology workflows actually need.
>
> I'd love to bring the same rigor — model selection by evidence, full
> production pipeline, explainability as a default — to Flatiron's
> oncology ML work. I'm based in / available for NYC and finishing my MS
> in Applied Statistics at RIT.
>
> Best,
> Derek Lankeaux

---

## 5. Headway — Senior Staff DS, Bayesian Experimentation & Causal Inference (STRETCH)

- **JD link:** https://job-boards.greenhouse.io/headway/jobs/5751656004
- **Location:** NYC
- **Lead project:** LLM Ensemble Bias Detection
- **Supporting:** AI Safety Red-Team, Breast Cancer
- **Why this is a stretch:** JD asks for 12+ years. Your playbook anti-pattern
  list says skip Staff/Principal unless the JD explicitly invites MS-level
  applicants. This one does not. **Recommended action:** either (a) swap
  for a non-Staff DS Bayesian role at the same tier, or (b) apply anyway
  and treat it as a pipeline-builder (response unlikely; cost of submission
  is 5 min).
- **JD phrases to mirror (if you do apply):** "Bayesian experimentation",
  "causal inference", "observational", "decision-making under uncertainty",
  "noisy signal", "reusable analysis tools"

### Cover letter (paste-ready, if you submit)

> I'm reaching out about the Senior Staff DS role — I want to be upfront
> that I'm earlier-career (finishing my MS in Applied Statistics at RIT in
> 2026), but I'm submitting because the methodological description maps
> exactly to how I already work, and I think a conversation is worth your
> 15 minutes if Headway is open to it.
>
> The recent project I'd point to is an LLM-ensemble bias-detection study:
> 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings from GPT-4o,
> Claude-3.5, and Llama-3.2. The headline finding — that 3 of 5 publishers
> showed statistically significant directional bias (Friedman chi-squared
> = 42.73, p < 0.001) — only holds because the pipeline was built to defend
> it: Krippendorff's alpha = 0.84 across raters, 92% pairwise correlation,
> a PyMC partial-pooling hierarchical model with R-hat < 1.01, and 95% HDI
> credible intervals per publisher rather than point estimates. Bootstrap
> CIs flagged 12.3% of passages as high-uncertainty for expert review.
> That "produce a credible interval, not a verdict" habit is the bar I'd
> bring to Headway's decision-making-under-noise problems.
>
> Happy to talk about whether there's an earlier-career version of this
> role on your team, or to revisit in 2027 if the experience gap is the
> blocker.
>
> Best,
> Derek Lankeaux

---

## Submission Workflow (your part, ~10 min/role)

For each of the 5 roles above:

1. Open the JD link, save as PDF into this directory:
   `applications/2026-06-12/<company>_JD.pdf`
2. Open `Resume_Derek_Lankeaux.md`, reorder Projects section so the lead
   project sits at the top, regenerate to PDF.
3. Paste the cover letter from this packet into the portal's cover letter
   field (or upload as PDF if required).
4. Answer common form questions using the **Application-form answers**
   section above (re-use across all 5).
5. Submit, then append the row to `application_tracker.csv` (5 rows are
   pre-staged below — uncomment / mark as `submitted` once done).

## Pre-staged tracker rows (copy/paste into `application_tracker.csv`)

```
2026-06-12,Anthropic,Research Engineer - Alignment Science,AI Safety,Remote/SF/NYC,greenhouse.io/anthropic,Project1_RedTeam,https://job-boards.greenhouse.io/anthropic/jobs/4009165008,resume_v3_safety.pdf,Y,N,queued,submit today,Lead with 340x cost hook + RSP tie-in
2026-06-12,Anthropic,Fellows Program - AI Security,AI Safety,Remote,anthropic.com/careers,Project1_RedTeam,https://www.anthropic.com/careers/jobs/5030244008,resume_v3_safety.pdf,Y,N,queued,submit today,Designed for emerging talent; strongest on-paper fit
2026-06-12,Hugging Face,ML Engineer - LLM Evaluation,LLM Eval,US Remote,apply.workable.com/huggingface,Project1_RedTeam,https://apply.workable.com/huggingface/,resume_v3_eval.pdf,Y,N,queued,submit today,Lead with Open LLM Leaderboard angle; emphasize 850/hr throughput
2026-06-12,Flatiron Health,Machine Learning Engineer,Healthcare ML,NYC,builtin.com,Project3_BreastCancer,https://builtin.com/job/machine-learning-engineer/4507605,resume_v3_healthcare.pdf,Y,N,queued,submit today,Lead with 99.12% acc + IEEE 2830-2025 tie-in
2026-06-12,Headway,Senior Staff DS - Bayesian Experimentation,DS Bayesian,NYC,greenhouse.io/headway,Project2_BiasDetection,https://job-boards.greenhouse.io/headway/jobs/5751656004,resume_v3_stats.pdf,Y,N,stretch,submit or sub,12+ yrs req'd; consider swapping for non-Staff DS role
```
