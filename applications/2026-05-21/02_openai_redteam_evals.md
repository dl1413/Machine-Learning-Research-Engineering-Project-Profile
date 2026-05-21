# OpenAI — Research Engineer, Red-Team / Dangerous-Capability Evals (NYC)

- **Role family:** AI Safety / Red-Team
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework
- **Supporting:** P2 — LLM Bias Detection · P3 — Breast Cancer Classification (rigor)
- **Source:** openai.com/careers (apply direct, Tier A)
- **Resume version:** `resume_v3_safety` (Projects section reordered → P1 first)

## JD KEYWORDS TO ECHO (fill from live JD before submit)
`__________`, `__________`, `__________`
Bank: red-teaming, jailbreak detection, harm taxonomy, eval harness,
model behavior, autograder, reliability.

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation** — *AI Safety / Red-Team* bullets
2. LLM Ensemble Bias Detection — *LLM Eval / Research Engineer* bullets
3. Breast Cancer Classification — *Generalist MLE* bullets

## Cover letter

Dear OpenAI Hiring Team,

I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5, Llama-3.2 —
that auto-grades 12,500 response pairs at 96.8% accuracy and 850 samples/hr,
with circuit breakers, async batching, and MLflow tracking baked in. Cost per
sample landed at $0.018, a 340x reduction versus human review.

The part most relevant to a red-team / dangerous-capability eval role is the
stacking layer: a 47-feature meta-classifier that reconciles disagreement
between the three judges across 6 harm categories (jailbreak, refusal-evasion,
policy-violation signals) and surfaces per-model blind spots via a PyMC
Bayesian hierarchical model with 95% HDI risk intervals. The whole pipeline
ships under IEEE 2830-2025 audit-trail requirements, reaching 97.2% precision
and ROC-AUC 0.9923.

Two supporting projects reinforce the fit. My LLM textbook-bias study pushed
the same ensemble to 67,500 ratings over 4,500 passages (Krippendorff's
alpha = 0.84) and localized publisher bias at p < 0.001 with post-hoc Nemenyi
tests — eval rigor at scale. A clinical-grade breast-cancer classifier (99.12%
accuracy, 100% precision, zero false positives, ROC-AUC 0.9987) shows the same
standard applied where errors carry real-world cost.

I'm based in / available for New York City and open to remote, targeting a 2026
start once I wrap my MS in Applied Statistics at RIT. Portfolio, code, and the
three technical reports are on my GitHub (dl1413). Happy to walk through the
red-team eval — it's the fastest way to see how I approach OpenAI's eval work.

Best,
Derek Lankeaux

## Pre-submit checklist
- [ ] 3+ JD phrases echoed in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio links live
- [ ] Work auth: Authorized to work in the US
- [ ] Salary: open, targeting market for role/location
- [ ] JD PDF saved in this folder
