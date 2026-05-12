# Application Packet — 2026-05-12 (Tue)

**Owner:** Derek Lankeaux
**Cadence:** 5 tailored applications, NYC + remote
**Lead-project distribution today:** Red-Team x2, Bias Detection x2, Breast Cancer x1

> Status note: each row below is *staged* — I (Claude) cannot submit to
> external ATS systems. Open each posting, paste the matching cover-letter
> intro and resume bullets from `APPLICATION_SNIPPETS.md`, submit, then flip
> the row's `status` in `application_tracker.csv` from `drafted` to
> `submitted` and save the JD PDF here under `applications/2026-05-12/`.

---

## App 1 — Anthropic · Research Engineer, Evaluations · Remote

- **Lead project:** AI Safety Red-Team Evaluation (Project 1)
- **Supporting:** LLM Bias Detection (Project 2), Breast Cancer (Project 3 — rigor signal)
- **Snippet sections:** P1 / "AI Safety / Red-Team / Alignment" bullets + "AI Safety lab" cover paragraph
- **Why this role:** Anthropic's evals team builds the exact harness pattern I shipped (multi-judge LLM + meta-classifier + Bayesian uncertainty). Constitutional AI keyword from the JD is already in the snippet pack.
- **Resume bullet order:** P1 -> P2 -> P3
- **JD keywords to echo verbatim:** "red-team", "evaluation harness", "Constitutional AI", "LLM-as-judge"
- **Cover-letter opener (paste-ready):**
  > Anthropic's evals work is where I want to be applying the harness I just
  > shipped. My independent AI Safety Red-Team Evaluation framework ensembles
  > GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking
  > classifier on 47 harm-signal features, reaching **96.8% accuracy and
  > ROC-AUC 0.9923** against a 12,500-pair benchmark across six harm
  > categories. It runs at 850 samples/hr for $0.018/sample — a **340x cost
  > reduction** versus human annotation — while holding Krippendorff's
  > alpha = 0.81 across raters. A PyMC Bayesian hierarchical layer adds 95%
  > HDI risk intervals per judge, and the whole pipeline ships under
  > IEEE 2830-2025 audit-trail requirements.

---

## App 2 — Scale AI · LLM Red-Team / Eval Engineer · Remote

- **Lead project:** AI Safety Red-Team Evaluation (Project 1)
- **Supporting:** LLM Bias Detection (Project 2)
- **Snippet sections:** P1 / "LLM Evaluation / Eval Infra" bullets + "Eval platform" cover paragraph
- **Why this role:** Scale's product *is* eval throughput at human-comparable quality. The 340x cost / 850 samples/hr / alpha = 0.81 triad is a direct map to their KPIs.
- **Resume bullet order:** P1 (eval-infra variant) -> P2 -> P3
- **JD keywords to echo verbatim:** "evaluation pipeline", "throughput", "inter-rater reliability", "MLOps"
- **Cover-letter opener (paste-ready):**
  > I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
  > Llama-3.2 — that auto-grades 12,500 response pairs at **96.8% accuracy
  > and 850 samples/hr**, with circuit breakers, async batching, and MLflow
  > tracking baked in. Cost per sample landed at $0.018, a 340x reduction
  > versus human review. The interesting part for Scale: the stacking
  > layer — a 47-feature meta-classifier that reconciles disagreement
  > between the three judges and surfaces per-model blind spots via Bayesian
  > hierarchical modeling. That maps directly to the eval-throughput +
  > quality problem you're solving for foundation-model customers.

---

## App 3 — Two Sigma · ML Researcher (Inference / Bayesian) · NYC

- **Lead project:** LLM Bias Detection (Project 2)
- **Supporting:** Breast Cancer (Project 3), AI Safety Red-Team (Project 1)
- **Snippet sections:** P2 / "Data Scientist (Bayesian / Causal)" bullets + "Data / Stats-heavy" cover paragraph
- **Why this role:** Two Sigma rewards full-Bayesian workflows and rigorous post-hoc testing. The Friedman -> Nemenyi -> hierarchical PyMC pipeline reads like one of their internal research notebooks.
- **Resume bullet order:** P2 -> P3 -> P1
- **JD keywords to echo verbatim:** "Bayesian", "hierarchical", "MCMC", "hypothesis testing", "PyMC"
- **Cover-letter opener (paste-ready):**
  > One project I'd point to is an LLM-ensemble bias-detection study I ran
  > last quarter: 4,500 textbook passages, 2.5M tokens, 67,500 LLM ratings
  > from GPT-4o / Claude-3.5 / Llama-3.2. The headline finding — that 3 of
  > 5 publishers showed statistically significant directional bias
  > (**Friedman chi-squared = 42.73, p < 0.001**) — only holds because the
  > pipeline was built to defend it: Krippendorff's alpha = 0.84 across
  > raters, 92% pairwise correlation, and a PyMC partial-pooling
  > hierarchical model with R-hat < 1.01 producing 95% HDI credible
  > intervals per publisher. That Bayesian-first habit is what I'd want
  > to bring to Two Sigma's inference work.

---

## App 4 — Memorial Sloan Kettering · Computational Oncology ML Engineer · NYC

- **Lead project:** Breast Cancer Classification (Project 3)
- **Supporting:** AI Safety Red-Team (Project 1 — for IEEE 2830-2025 / audit rigor), LLM Bias Detection (Project 2)
- **Snippet sections:** P3 / "Healthcare / Clinical ML" bullets + "Healthcare ML" cover paragraph
- **Why this role:** MSK Computational Oncology cares about clinical-grade calibration, SHAP transparency, and zero-FP screening — all explicit in the Project 3 snippet pack.
- **Resume bullet order:** P3 -> P1 -> P2
- **JD keywords to echo verbatim:** "clinical", "calibration", "explainability", "FDA / IEEE 2830-2025", "SHAP"
- **Cover-letter opener (paste-ready):**
  > The work most relevant to MSK Computational Oncology is a clinical-grade
  > breast-cancer classifier I shipped this year. I benchmarked 8 algorithms
  > end-to-end — Random Forest through stacking ensembles — and landed at
  > **99.12% accuracy with 100% precision (zero false positives), 98.59%
  > recall, and ROC-AUC 0.9987**, comfortably above the 90-95% range
  > typically cited for human expert reads. Just as important for clinical
  > deployment: the pipeline ships with SHAP explanations per prediction,
  > Platt-calibrated probabilities (ECE 0.0089), VIF-pruned features, and a
  > FastAPI service under 100ms p95, all aligned with IEEE 2830-2025
  > transparency standards.

---

## App 5 — Hugging Face · Research Engineer, Evaluations · NYC + Remote

- **Lead project:** LLM Bias Detection (Project 2)
- **Supporting:** AI Safety Red-Team (Project 1), Breast Cancer (Project 3)
- **Snippet sections:** P2 / "LLM Eval / Research Engineer" bullets + "LLM platform / eval startup" cover paragraph
- **Why this role:** Hugging Face's evals team owns reliability + reproducibility of public benchmarks. The 67,500-rating / R-hat < 1.01 / MLflow lineage story is the exact pitch.
- **Resume bullet order:** P2 -> P1 -> P3
- **JD keywords to echo verbatim:** "benchmark", "reproducibility", "open-source", "LLM evaluation"
- **Cover-letter opener (paste-ready):**
  > I'm interested in Hugging Face because I've spent the last few months
  > building exactly the kind of multi-LLM evaluation infrastructure your
  > evals team productionizes for the community. My textbook-bias study ran
  > **67,500 ratings through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble**
  > over 4,500 passages, with circuit-breakered async API integration and
  > full MLflow lineage end-to-end. The eval rubric held Krippendorff's
  > alpha = 0.84 and surfaced publisher-level bias at p < 0.001 — the kind
  > of reliability scaffolding that lets a public benchmark survive scrutiny.
  > I'd love to help build that scaffolding into HF Evaluate.

---

## Universal Closer (paste at end of every cover letter)

> I'm based in / available for New York City and open to remote, targeting
> a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
> code, and the three technical reports referenced above are on my GitHub
> (dl1413). Happy to walk through any of them — the red-team eval is
> probably the fastest way to see how I think about [Company's problem].
>
> Best,
> Derek Lankeaux

---

## Today's Submit Checklist

- [ ] App 1 — Anthropic — submit, save JD PDF, flip tracker row to `submitted`
- [ ] App 2 — Scale AI — submit, save JD PDF, flip tracker row to `submitted`
- [ ] App 3 — Two Sigma — submit, save JD PDF, flip tracker row to `submitted`
- [ ] App 4 — MSK — submit, save JD PDF, flip tracker row to `submitted`
- [ ] App 5 — Hugging Face — submit, save JD PDF, flip tracker row to `submitted`

**Reminder (from playbook §5):** every app must echo 3+ JD phrases verbatim,
include a metric hook in the opener, and pass the work-auth / salary
expectation fields with the standard answers.
