# 01 — Capital One: Principal Associate, Data Scientist — LLM Customization Team

**Location:** New York, NY (hybrid)
**Job link:** https://www.capitalonecareers.com/job/new-york/principal-associate-data-scientist-llm-customization-team/1732/92083762528
**Alt link (Workday):** https://capitalone.wd12.myworkdayjobs.com/en-US/Capital_One/job/New-York-NY/Principal-Associate--Data-Scientist---LLM-Customization-Team_R229942-2/apply/applyManually
**Verified:** Posting was live as of the search on 2026-07-07 — confirm before applying.

## What they want (pulled from posting)

> "Be an expert in NLP to harness the power of LLMs, adapting and finetuning them for business-specific applications, and building NLP models through all phases of development from design through training, evaluation, and validation."
> Required: hands-on LLMs, open-source tools, cloud, RLHF / self-supervised / explainability / training optimization.

## Why this candidate fits (evidence, in one line each)

- **LLM adaptation & evaluation at production scale** — built 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) processing 2.5M tokens with MLflow tracking and 92% pairwise agreement (LLM Bias Detection).
- **Validation methodology** — Krippendorff's α = 0.81–0.84, Bayesian hierarchical uncertainty (R-hat < 1.01, 95% HDI), 47 engineered features, SHAP explainability (AI Safety Red-Team).
- **Regulated-industry sensibility** — IEEE 2830-2025 / ISO/IEC 23894 / EU AI Act artifacts; useful for a bank that has to defend model decisions to regulators (Breast Cancer ML calibration story: ECE reduced 71.5%).

## Project → Requirement mapping

| Capital One requirement | Anchor project | One-line evidence |
|---|---|---|
| Finetuning / adapting LLMs | AI Safety Red-Team | Dual-stage ensemble → ML classifier, 96.8% acc, 340× cheaper than human labels |
| Evaluation & validation | LLM Bias Detection | α = 0.84, χ² = 42.73 (p < 0.001), publisher-level HDI |
| Explainability / RLHF-adjacent | AI Safety Red-Team | SHAP over 47 features, Constitutional-AI-style safety filter |
| Training optimization | Breast Cancer ML | Optuna TPE (45 vs 240 trials), Platt calibration |
| Production MLOps | All 3 | MLflow, FastAPI <100ms p95, circuit breakers |

## Cover letter (paste-ready)

> Dear Capital One AI Foundations team,
>
> I am applying for the Principal Associate, Data Scientist — LLM Customization role because the team's mandate to "harness LLMs for business-specific applications" is exactly the loop I have been running independently for the past year, and I would like to run it against real customer problems in a regulated environment.
>
> My AI Safety Red-Team Evaluation project built a dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that annotates responses, then trains a Stacking Classifier on 47 engineered features to predict harm. It processes 850 samples/hour, hits 96.8% accuracy with Krippendorff's α = 0.81, and reduces annotation cost 340× ($6.12 → $0.018 per sample). The same pattern — multi-LLM label → calibrated classifier — is directly reusable for adapting LLMs to Capital One's tasks where human labels are expensive and audit trails matter.
>
> My LLM Ensemble Bias Detection work is a closer fit to the "validation" part of the JD: 67,500 ratings across 4,500 passages with a PyMC hierarchical model (R-hat < 1.01, 95% HDI) that quantifies publisher-level bias. The uncertainty-first framing is what a bank needs to defend a model output.
>
> My Breast Cancer ML Classification project (99.12% accuracy, 100% precision, ECE reduced 71.5% via Platt scaling) shows I can also carry a supervised model end-to-end with calibration and threshold policies — the mechanical work behind any production LLM feature.
>
> All three projects ship with MLflow tracking, SHAP explainability, and IEEE 2830-2025-aligned model cards. I am authorized to work in the US, based in the Northeast, and available for a 2026 start.
>
> Thank you for the consideration — I would welcome a conversation.
>
> Derek Lankeaux
> linkedin.com/in/derek-lankeaux · github.com/dl1413

## Screener prep

- Expect: "walk me through how you'd finetune a small model for a Capital One customer-service task." Answer with LoRA + eval harness structure from Red-Team project.
- Expect: "how do you catch regressions?" → paired α + Friedman χ² pattern from Bias Detection.
- Comp expectation: NYC L4 range typically $175–225k base + equity — Capital One posts specific bands per JD, confirm on the app screen.

## Resume tweaks for this app

- Move "340× cost reduction via GenAI" into the top 2 bullets of the summary.
- Add a line under LLM Bias Detection explicitly mentioning "publisher-level credible intervals" — mirrors banking risk-tier framing.
