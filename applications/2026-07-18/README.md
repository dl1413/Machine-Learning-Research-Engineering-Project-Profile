# Daily Job Applications — 2026-07-18

**Candidate:** Derek Lankeaux, MS (Applied Statistics)
**Portfolio anchor:** [dl1413.github.io/LLM-Portfolio](https://dl1413.github.io/LLM-Portfolio/)
**Location filter:** New York City or Remote

## The 3 primary projects referenced across every application

1. **AI Safety Red-Team Evaluation Framework** — dual-stage LLM ensemble (GPT-4o + Claude-3.5 + Llama-3.2) + Stacking Classifier. 96.8% accuracy, Krippendorff's α = 0.81, 340× cost reduction ($0.018 vs $6.12/sample), 12,500 response pairs, 6 harm categories, Bayesian hierarchical risk model (95% HDI), SHAP, IEEE 2830-2025.
2. **LLM Ensemble Textbook Bias Detection** — 67,500 bias ratings across 4,500 passages, 2.5M tokens. Krippendorff's α = 0.84, Friedman χ² = 42.73 (p < 0.001), PyMC hierarchical model with partial pooling (R-hat < 1.01), 3/5 publishers with credible bias.
3. **Clinical-Grade Breast Cancer ML Classification** — AdaBoost ensemble. 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987. Platt calibration (ECE 0.0089), SHAP, MLflow registry, FastAPI (<100ms p95), IEEE 2830-2025.

## Today's 5 targets

| # | Company | Role | Location | Directory |
|---|---------|------|----------|-----------|
| 1 | Anthropic | Research Scientist, Frontier Red Team (Cyber) | Remote / SF | `01_anthropic_frontier_red_team/` |
| 2 | Uber | Senior Applied Scientist, AI Red Teaming & Model Risk | New York, NY | `02_uber_ai_red_teaming/` |
| 3 | Headway | Senior Staff Data Scientist — Bayesian Experimentation & Causal Inference | NYC / Remote | `03_headway_bayesian_experimentation/` |
| 4 | Capital One | Principal Associate, Data Scientist — LLM Customization Team | New York, NY | `04_capitalone_llm_customization/` |
| 5 | Dataminr | Senior Research Scientist (NLP, LLM, GenAI) | Remote (US) | `05_dataminr_senior_research_scientist/` |

Each folder contains:
- `job.md` — role snapshot, requirements, source URL
- `cover_letter.md` — tailored cover letter you can paste or attach
- `project_mapping.md` — bullet-by-bullet map from the 3 projects to their JD
- `resume_notes.md` — which resume sections / bullets to emphasize

## How to use this batch

1. Open `job.md` in each folder, confirm the requisition is still open.
2. Skim `project_mapping.md` — copy relevant bullets into the online form's short-answer fields.
3. Use `cover_letter.md` as the attached cover letter (edit the first paragraph if you learn something recent about the team).
4. Apply. Mark the folder DONE by appending `_APPLIED_<yyyy-mm-dd>` to the directory name, or just log the timestamp in a follow-up commit.

## Why these 5

- **Anthropic FRT** and **Uber AI Red Teaming** hit the AI Safety Red-Team project head-on (dual-stage LLM ensemble, harm taxonomy, reusable eval pipelines).
- **Headway** is the cleanest match for the Bayesian hierarchical modeling / experimentation half of the profile (PyMC, MCMC, causal inference, A/B).
- **Capital One LLM Customization** exercises the LLM-as-judge + production LLM ensembling half (three-model correlation, prompt iteration, MLflow tracking).
- **Dataminr** rounds it out with a research-oriented NLP/LLM/GenAI seat — publications-friendly, and the technical reports in this repo double as writing samples.
