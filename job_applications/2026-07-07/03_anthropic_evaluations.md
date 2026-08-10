# 03 — Anthropic: Research Engineer — Model Evaluations / Safety

**Location:** New York, NY (also SF, Seattle, and remote-eligible depending on posting)
**Careers portal:** https://www.anthropic.com/careers/jobs
**Verified:** Anthropic careers page confirmed live 2026-07-07. Filter for "Evaluations", "Safety", "Red Team", or "Alignment" — role titles rotate. Fellows Program (May & July 2026 cohorts) is also open: https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/

## What they want (typical evals-role JD)

> Build evaluation suites that catch hallucinations, regressions, bias, and grounding gaps before production. Production experience with LLMs including advanced prompt engineering, agent development, evaluation frameworks, and deployment at scale.

## Why this is the single strongest fit of the day

Derek's **AI Safety Red-Team Evaluation** project is literally what this team ships. It maps 1:1:

- Dual-stage LLM ensemble → ML classifier for automated harm detection
- 12,500 AI response pairs across 6 harm categories, α = 0.81
- 8-category MITRE ATLAS-aligned adversarial taxonomy
- Multi-turn escalation identified as highest-risk vector (31.8%)
- Defense analysis: dual-filter reduces harm 21.8% → 4.8% (78% reduction)
- IEEE 2830-2025 / EU AI Act artifacts

## Project → Requirement mapping

| Anthropic evals requirement | Anchor project | Evidence |
|---|---|---|
| Build eval harnesses that catch regressions | AI Safety Red-Team | 47-feature classifier, MLflow-tracked, SHAP audits |
| Adversarial / red-team taxonomy | AI Safety Red-Team | 8-category MITRE ATLAS-aligned framework |
| Inter-rater reliability across models | LLM Bias Detection | α = 0.84 across 3 frontier LLMs, 92% pairwise |
| Bayesian uncertainty quantification | LLM Bias Detection | PyMC hierarchical, R-hat < 1.01, 95% HDI |
| Calibrated decision policies | Breast Cancer ML | Platt scaling → ECE 0.0312 → 0.0089, threshold tuning for context |

## Cover letter

> Dear Anthropic Evaluations team,
>
> I am applying because my portfolio's centerpiece is the exact problem the Evaluations team owns. My AI Safety Red-Team Evaluation project built a dual-stage pipeline — LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) generates labels, a Stacking Classifier over 47 engineered features learns to predict harm — and evaluated 12,500 AI response pairs across 6 harm categories with Krippendorff's α = 0.81 and 96.8% accuracy. The 340× cost reduction versus human annotation ($6.12 → $0.018 per sample) is the ROI story that lets an eval program scale.
>
> The project catalogs adversarial attacks via an 8-category MITRE ATLAS-aligned taxonomy and identifies multi-turn escalation as the highest-risk vector (31.8% of successful attacks). It also quantifies defense effectiveness: a dual-filter setup reduces harm rate from 21.8% to 4.8%, a 78% reduction. Bayesian hierarchical modeling in PyMC (R-hat < 1.01, 95% HDI) turns per-model risk into decision-ready credible intervals.
>
> My LLM Ensemble Bias Detection work extends the same reliability toolkit to a document-scale evaluation: 67,500 ratings, 92% pairwise LLM correlation, publisher-level bias detection with Friedman χ² = 42.73 (p < 0.001). And my Breast Cancer ML Classification project (99.12% acc, ECE reduced 71.5% via Platt scaling) shows I understand calibration and threshold policies — the difference between a metric and a shipped decision.
>
> All three ship with MLflow, SHAP, and IEEE 2830-2025 / ISO 23894-aligned model cards. I have an MS in Applied Statistics from RIT (2026), am authorized to work in the US, and available for both the Research Engineer track and the Fellows Program.
>
> Thank you for reading — I would welcome a screening call.
>
> Derek Lankeaux · linkedin.com/in/derek-lankeaux · github.com/dl1413

## Screener prep

- Expect: "walk me through your Red-Team eval pipeline end-to-end." Have the architecture diagram and the failure-mode analysis (multi-turn escalation) ready.
- Expect: coding — implement a small eval harness with pairwise-agreement statistic. Practice α + Cohen's κ from memory.
- Expect: safety-values / mission fit. Read the Responsible Scaling Policy before interview.

## Resume tweaks

- Top-line summary should lead with the Red-Team project — this is Anthropic's dead center.
- Add "multi-turn escalation risk analysis" as an explicit bullet — Anthropic's evals literature calls this out.
- Cross-link the Fellows Program in your cover — signals you know both tracks exist.
