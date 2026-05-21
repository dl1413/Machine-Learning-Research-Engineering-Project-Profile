# Anthropic — Research Engineer, Evaluations (Remote / NYC)

- **Role family:** AI Safety / Eval Engineering
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework
- **Supporting:** P2 — LLM Bias Detection · P3 — Breast Cancer Classification (rigor)
- **Source:** careers.anthropic.com (apply direct, Tier A)
- **Resume version:** `resume_v3_safety` (Projects section reordered → P1 first)

## JD KEYWORDS TO ECHO (fill from live JD before submit)
`__________`, `__________`, `__________`
Bank to draw on: red-teaming, LLM-as-judge, eval harness, harm classification,
Constitutional AI, inter-rater reliability, audit trail.

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation** — use the *AI Safety / Red-Team* bullet set
2. LLM Ensemble Bias Detection — *LLM Eval / Research Engineer* bullets
3. Breast Cancer Classification — *Generalist MLE* bullets (signals rigor)

## Cover letter

Dear Anthropic Hiring Team,

Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that detects
harmful AI outputs at 96.8% accuracy and 340x lower cost than human annotation,
while preserving audit-grade reliability (Krippendorff's alpha = 0.81).

My most relevant work for Anthropic's mission is an independent AI Safety
Red-Team Evaluation framework I built and published in January 2026. It
ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and
ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories. The
pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost reduction
versus human annotation — while holding inter-rater reliability at
Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian hierarchical
model that produces 95% HDI risk intervals per judge, and shipped the whole
thing under IEEE 2830-2025 audit-trail requirements.

Two supporting projects show the same eval-first instinct at scale and under
stakes. My LLM textbook-bias study ran 67,500 ratings across 4,500 passages
(Krippendorff's alpha = 0.84) and used a partial-pooling hierarchical model
(R-hat < 1.01) to defend a publisher-bias finding at p < 0.001 — exactly the
"don't claim it unless the reliability scaffolding holds" discipline evals
demand. And a clinical-grade breast-cancer classifier (99.12% accuracy, 100%
precision, ROC-AUC 0.9987) shows I hold that bar even when a false call has
real cost.

I'm based in / available for New York City and open to remote, targeting a 2026
start once I wrap my MS in Applied Statistics at RIT. Portfolio, code, and the
three technical reports referenced above are on my GitHub (dl1413). The
red-team eval is probably the fastest way to see how I think about Anthropic's
evaluation work.

Best,
Derek Lankeaux

## Pre-submit checklist
- [ ] 3+ JD phrases echoed in resume + cover letter
- [ ] LinkedIn / GitHub / Portfolio links live
- [ ] Work auth: Authorized to work in the US
- [ ] Salary: open, targeting market for role/location
- [ ] JD PDF saved in this folder
