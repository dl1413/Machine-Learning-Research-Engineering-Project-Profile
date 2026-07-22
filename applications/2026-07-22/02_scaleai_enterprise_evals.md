# Scale AI — AI Research Engineer, Enterprise Evaluations

- **Location:** New York, NY (or San Francisco)
- **Lead project:** AI Safety Red-Team Evaluation
- **Supporting projects:** LLM Ensemble Bias Detection, Breast Cancer Classification
- **JD:** https://job-boards.greenhouse.io/scaleai/jobs/4629589005
- **Portfolio:** https://github.com/dl1413/machine-learning-research-engineering-project-profile

---

## Cover Letter

Hi Scale AI Enterprise Evaluations team,

The AI Research Engineer opening on Enterprise Evaluations reads like a
scaled-up version of the pipeline I've built and shipped three times
over the last year: multi-model LLM ensemble that grades responses at
production volume, with inter-rater reliability that lets enterprise
customers actually trust the numbers.

The direct fit is my **AI Safety Red-Team Evaluation** framework — a
dual-stage system (GPT-4o + Claude-3.5 + Llama-3.2 → stacking
classifier) that evaluated **12,500 response pairs across 6 harm
categories at 96.8% accuracy, α = 0.81, and 850 samples/hour**. Cost
per graded sample dropped from **$6.12 (human) to $0.018 (LLM
ensemble) — 340× — with no loss of audit-grade reliability**. The
adversarial taxonomy leans on MITRE ATLAS; the defense analysis
showed dual-filter reducing harm rate from 21.8% → 4.8% (78%
reduction). Every number is tied to a published, reproducible report.

The Enterprise angle — where an eval has to hold up not just against
research reviewers but against a customer's compliance function —
shows up in two follow-on projects:

- **LLM Bias Detection**: 67,500 ratings, 92% pairwise correlation
  across frontier LLMs, Bayesian hierarchical publisher-level
  detection, Friedman χ² = 42.73 (p < 0.001). Shipped as a technical
  report with 95% HDI intervals and reproducibility artifacts.
- **Breast Cancer Classification**: 99.12% accuracy, Platt-calibrated
  (ECE 0.0089), threshold-optimized for context-specific decision
  policies. It's a regulated-domain project by design — SHAP + fairness
  audit + IEEE 2830-2025 model card in the same repo.

If Enterprise Evaluations is about turning "we ran an eval" into "we
ran an eval that survives a customer's legal review," that's the shape
of work I've been optimizing for.

Thanks for your time. Full portfolio and reports at the link above.

Best,
Derek Lankeaux
MS Applied Statistics, Rochester Institute of Technology (2026)
LinkedIn: https://linkedin.com/in/derek-lankeaux

---

## One-Page Pitch

Enterprise eval is a measurement + reliability + defensibility problem
in a trench coat. My red-team framework already hits the middle piece
(α = 0.81 across three frontier LLMs, 92% pairwise correlation on a
different dataset in a follow-on project). The reliability half comes
from the Bayesian layer — hierarchical partial pooling with MCMC
diagnostics (R-hat < 1.01, ESS reported) so per-model risk gets
uncertainty-quantified instead of averaged into a lie. The
defensibility half comes from the artifacts: IEEE 2830-2025 model
card, ISO/IEC 23894:2025 risk register, EU AI Act mapping, published
technical report, SHAP-based explanations. I've done all three at
80K+ API calls / 2.5M tokens with circuit breakers and MLflow
tracking, so the pipeline runs, and the runs get logged. That's the
starter kit for taking one of your customer eval suites from
"deployed" to "auditable."

---

## Follow-up plan

- **T+0:** Submit via greenhouse; if NYC/SF is a choice, pick NYC.
- **T+3 days:** Referral hunt on LinkedIn — Scale has a lot of alumni
  in NYC. Filter for MS/PhD Statistics, message 2-3.
- **T+7 days:** If a recruiter reaches out, ask about team split and
  whether the role is more platform (eval framework) or applied
  (customer-facing evals). The answer changes which two supporting
  projects to lead with in the phone screen.
