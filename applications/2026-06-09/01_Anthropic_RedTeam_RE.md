# Anthropic — Research Engineer, Frontier Red Team

**Location:** Remote (US) with NYC presence · **Lead project:** Project 1 (AI Safety Red-Team Evaluation)
**Source:** careers.anthropic.com · **Snippet base:** APPLICATION_SNIPPETS §1 (Safety / Red-Team)

## JD keyword echo (≥3 verbatim)
- "red-teaming", "model evaluations", "harm taxonomy", "Constitutional AI", "policy violation"

## Resume Projects-section order
1. **AI Safety Red-Team Evaluation Framework** (LEAD)
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer ML Classification

## Tailored resume bullets (top of Projects)
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) red-teaming 12,500 response pairs across 6 harm categories at **96.8% accuracy, ROC-AUC 0.9923**, Krippendorff's alpha = 0.81
- Cut red-team annotation cost **340x ($6.12 -> $0.018/sample)** at 850 samples/hr throughput on circuit-breakered async pipeline
- Built 8-category MITRE ATLAS-aligned attack taxonomy; multi-turn escalation surfaced as highest-risk vector (31.8% of confirmed harms)
- Quantified dual-filter defense effectiveness: harm rate 21.8% -> 4.8% (78% reduction); reported per-model 95% HDI via PyMC hierarchical model
- Shipped IEEE 2830-2025-compliant audit trail with SHAP attribution per prediction

## Cover letter

> Dear Anthropic Frontier Red Team,
>
> My most relevant work for this role is an independent AI Safety Red-Team
> Evaluation framework I published in April 2026. It ensembles GPT-4o,
> Claude-3.5-Sonnet, and Llama-3.2 as red-team judges and trains a stacking
> classifier on 47 harm-signal features, reaching **96.8% accuracy and
> ROC-AUC 0.9923** against a 12,500-pair benchmark spanning 6 harm
> categories (dangerous info, hate, deception, privacy, illegal activity,
> self-harm). The pipeline runs at 850 samples/hr for $0.018/sample — a
> **340x cost reduction** versus human annotation — while holding
> inter-rater reliability at **Krippendorff's alpha = 0.81**. I paired it
> with a PyMC Bayesian hierarchical model producing 95% HDI risk intervals
> per judge, and an 8-category MITRE ATLAS-aligned attack taxonomy that
> flagged multi-turn escalation as the highest-risk vector (31.8%).
>
> Two adjacent projects show the breadth: an LLM-ensemble bias study
> (67,500 ratings, Friedman chi-squared = 42.73, p < 0.001) and a
> clinical-grade classifier (99.12% accuracy, zero false positives, FastAPI
> at <100ms p95) — both shipped end-to-end with MLflow, model cards, and
> reproducibility scaffolding aligned with IEEE 2830-2025 and ISO/IEC
> 23894:2025.
>
> I'm finishing my MS in Applied Statistics at RIT (2026), authorized to
> work in the US, open to remote or NYC. I'd love to bring this combination
> of eval rigor and production throughput to Frontier Red Team.
>
> Best,
> Derek Lankeaux
