# Daily Application Plan — Fri 2026-05-15

**Target:** 5 applications, NYC + Remote. Every app uses all 3 primary projects
(1 lead + 2 supporting). Role-family spread chosen so we don't pile all five
into one bucket and starve the others of data.

| # | Company | Role | Loc | Tier | Lead Project | Supporting |
|---|---------|------|-----|------|--------------|------------|
| 1 | Anthropic | Research Engineer, Alignment | Remote/NYC | A | P1 Red-Team | P2, P3 |
| 2 | Hugging Face | Evaluation Research Engineer | Remote | B | P1 Eval | P2, P3 |
| 3 | Memorial Sloan Kettering | Computational Oncology ML Engineer | NYC | D | P3 Breast Cancer | P1, P2 |
| 4 | Two Sigma | ML Research Engineer (Modeling) | NYC | C | P2 Bayesian | P1, P3 |
| 5 | Patronus AI | LLM Eval Platform Engineer | NYC + Remote | B | P1 Eval Infra | P2, P3 |

Coverage check: P1 leads 3x, P2 leads 1x, P3 leads 1x — and all 3 appear in
every packet. Matches the playbook rule "lead with best match, list other two
as supporting evidence."

---

## Packet 1 — Anthropic, Research Engineer (Alignment)

**Resume version:** `resume_v3_safety.pdf` (Project 1 at top of Projects)

**Cover-letter opener (paste verbatim):**

> Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that detects
> harmful AI outputs at **96.8% accuracy and 340x lower cost** than human
> annotation, while preserving audit-grade reliability (Krippendorff's
> alpha = 0.81).

**Body paragraph:** Use APPLICATION_SNIPPETS.md §"Cover letter paragraph —
AI Safety lab" verbatim. Substitute `[Company]` -> `Anthropic`,
`[specific team/product]` -> `the Alignment team's Constitutional AI and
red-teaming work`.

**Bridge to supporting projects (one sentence):**

> The same evaluation discipline shows up in two adjacent projects on my
> portfolio: a 67,500-rating LLM bias study (Krippendorff alpha = 0.84,
> Friedman chi-squared = 42.73, p < 0.001) and a clinical-grade ensemble
> classifier (99.12% accuracy, 100% precision) shipped behind a sub-100ms
> FastAPI service.

**JD keywords to mirror:** alignment, red-teaming, Constitutional AI,
eval harness, LLM-as-judge, model behavior, RLHF data quality.

**Submit via:** careers.anthropic.com (direct apply). Save JD PDF in this
folder.

---

## Packet 2 — Hugging Face, Evaluation Research Engineer

**Resume version:** `resume_v3_safety.pdf` (Project 1 lead)

**Cover-letter opener:**

> Shipped a 3-model LLM eval harness (GPT-4o, Claude-3.5, Llama-3.2) auto-grading
> 12,500 response pairs at 96.8% accuracy and 850 samples/hr, with circuit
> breakers, async batching, and MLflow lineage end-to-end — $0.018 per sample.

**Body paragraph:** APPLICATION_SNIPPETS.md §"Cover letter paragraph — Eval
platform" verbatim. `[Company]` -> `Hugging Face`. Add one sentence on
open-source: "I'm a long-time HF Hub user (datasets, transformers, evaluate)
and the eval harness was built against the same primitives your community
relies on."

**Bridge:**

> The methodology generalized: I ran the same ensemble against 4,500
> textbook passages (2.5M tokens, 67,500 ratings) to surface publisher-level
> bias at p < 0.001, and used the eval-rigor habit on a clinical ML
> benchmark that landed at 99.12% accuracy / ROC-AUC 0.9987.

**JD keywords:** evaluation harness, LLM-as-judge, leaderboards, reproducible
benchmarks, open source, datasets, transformers, MLflow.

**Submit via:** apply.workable.com/huggingface/ (or careers page). Mention HF
username if you have one.

---

## Packet 3 — Memorial Sloan Kettering, Computational Oncology ML Engineer

**Resume version:** `resume_v3_clinical.pdf` (Project 3 at top of Projects)

**Cover-letter opener:**

> Trained an 8-algorithm benchmark ensemble that hits **99.12% accuracy,
> 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987**
> on breast-cancer classification — above the 90-95% human-expert range —
> and deployed it as a <100ms p95 FastAPI service with SHAP explanations
> per prediction.

**Body paragraph:** APPLICATION_SNIPPETS.md §"Cover letter paragraph —
Healthcare ML role" verbatim. `[Company]` -> `Memorial Sloan Kettering`,
`[Company's specific clinical-AI problem]` -> `the computational oncology
group's diagnostic ML work`.

**Bridge:**

> The same audit-first habit shows up in my LLM work: a published AI safety
> red-team eval (96.8% accuracy, IEEE 2830-2025 compliant, 340x cost
> reduction vs human annotation) and a Bayesian bias-detection study using
> PyMC partial-pooling hierarchical models (R-hat < 1.01, 95% HDI per
> publisher).

**JD keywords:** clinical decision support, SHAP, explainability,
HIPAA/PHI awareness, IEEE 2830, FastAPI, model registry, computational
pathology / oncology.

**Submit via:** mskcc.org/careers. Note MS-Applied-Statistics-in-progress.

---

## Packet 4 — Two Sigma, ML Research Engineer (Modeling)

**Resume version:** `resume_v3_research.pdf` (Project 2 at top, P1/P3 below)

**Cover-letter opener:**

> Fit a PyMC Bayesian hierarchical model with partial pooling across
> publishers on 67,500 LLM ratings (R-hat < 1.01, ESS > 1000) and produced
> 95% HDI credible intervals that supported a Friedman chi-squared = 42.73,
> p < 0.001 finding of publisher-level bias.

**Body paragraph:** APPLICATION_SNIPPETS.md §"Cover letter paragraph —
Data / Stats-heavy role" verbatim. `[Company]` -> `Two Sigma`, append:
"That Bayesian-first habit is what I'd want to bring to Two Sigma's
modeling research — preferring posterior distributions over point estimates,
and treating MCMC convergence diagnostics as a release gate."

**Bridge:**

> Two adjacent projects show the same statistical discipline applied to
> different domains: a 12,500-pair AI safety eval that landed at ROC-AUC
> 0.9923 with Krippendorff alpha = 0.81, and a clinical ML ensemble
> (99.12% accuracy, 100% precision) built under nested cross-validation.

**JD keywords:** Bayesian, MCMC, hierarchical, signal research, PyMC,
statistical rigor, hypothesis testing, Python, research engineer.

**Submit via:** twosigma.com/careers. Highlight RIT Applied Statistics
program.

---

## Packet 5 — Patronus AI, LLM Eval Platform Engineer

**Resume version:** `resume_v3_safety.pdf` (Project 1 lead, Project 2
strongly supporting)

**Cover-letter opener:**

> Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading
> 12,500 response pairs at 96.8% accuracy, 850 samples/hr, $0.018/sample
> (340x cheaper than human annotation), with a 47-feature meta-classifier
> reconciling judge disagreement.

**Body paragraph:** APPLICATION_SNIPPETS.md §"Cover letter paragraph —
Eval platform (Patronus / Galileo / Arize)" verbatim. `[Company]` ->
`Patronus AI`. Add: "Your product abstracts the exact pipeline I just
hand-rolled — circuit breakers, async batching, MLflow lineage, multi-judge
reconciliation — so the engineering map is short."

**Bridge:**

> Same eval infrastructure ran a 67,500-rating bias study end-to-end
> (Krippendorff alpha = 0.84, Friedman p < 0.001), and the production
> habits show up in my clinical-ML side: <100ms p95 FastAPI, SHAP per
> prediction, MLflow registry.

**JD keywords:** eval harness, LLM-as-judge, production eval, observability,
guardrails, hallucination detection, drift monitoring.

**Submit via:** patronus.ai/careers (or YC Work-at-a-Startup listing).

---

## Universal closer (all 5)

Paste APPLICATION_SNIPPETS.md §"Universal Cover Letter Closer" at the end
of each. Verify LinkedIn, GitHub (dl1413), portfolio URL all live before
hitting submit.

## Pre-submit checklist (per packet)

- [ ] Resume PDF rebuilt with lead project at top of Projects section
- [ ] >= 3 JD phrases appear verbatim in resume + cover letter
- [ ] Metric hook present in opener
- [ ] Salary expectation answered: "open, targeting market for the
      role/location"
- [ ] Work authorization answered
- [ ] JD saved as PDF in `applications/2026-05-15/<company>_jd.pdf`
- [ ] Row appended to `application_tracker.csv` (already pre-staged below)
