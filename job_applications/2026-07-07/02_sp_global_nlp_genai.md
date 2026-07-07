# 02 — S&P Global: Data Scientist — NLP, LLM & GenAI

**Location:** New York, NY
**Job link:** https://www.linkedin.com/jobs/view/data-scientist-%E2%80%93-nlp-llm-and-genai-at-s-p-global-3868183572
**Comp:** Posted $85k–$150k base (verify current)
**Verified:** Live per search on 2026-07-07 — confirm before applying.

## What they want

> "Hands-on entry-level and experienced ML and NLP scientists to apply technical expertise in NLP, deep learning, GenAI, and LLMs to drive business value."

Read between the lines: S&P Global runs a lot of financial-document NLP (10-Ks, ratings commentary, ESG). They want someone who can build LLM pipelines against messy long documents *and* defend the numbers to a compliance team.

## Why this candidate fits

- Multi-LLM ensemble experience over long-form structured text (**LLM Bias Detection** worked on 4,500 textbook passages — same "long documents, judged by LLMs" shape as SEC filings and ratings commentary).
- Statistical defensibility — Bayesian HDI, Krippendorff's α, multiple-testing correction — is exactly what S&P Global's compliance-adjacent DS work requires.
- Cost/latency framing (**340× reduction, <100ms p95**) is the language S&P leadership uses when discussing GenAI ROI internally.

## Project → Requirement mapping

| S&P requirement | Anchor project | Evidence |
|---|---|---|
| LLM evaluation on long-form documents | LLM Bias Detection | 4,500 passages × 3 LLMs × 5 dimensions; α = 0.84 |
| NLP / deep learning production pipelines | AI Safety Red-Team | 850 samples/hour, MLflow, circuit breakers |
| GenAI cost/latency awareness | AI Safety Red-Team | $0.018/sample vs $6.12 human; 340× reduction |
| Statistical rigor for regulated output | LLM Bias Detection | Friedman χ², PyMC HDI, R-hat diagnostics |
| Supervised modeling fallback | Breast Cancer ML | 99.12% acc, ROC-AUC 0.9987, Platt calibration |

## Cover letter

> Dear S&P Global NLP team,
>
> I am writing to apply for the Data Scientist — NLP, LLM & GenAI role. My independent work over the past year has focused on the same problem shape S&P deals with daily: extracting defensible signal from long documents when the ground truth is expensive to collect.
>
> My LLM Ensemble Textbook Bias Detection project processed 4,500 passages with a 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) and 2.5M tokens, then fit a PyMC hierarchical model (R-hat < 1.01, 95% HDI) to quantify publisher-level bias. The Friedman χ² of 42.73 (p < 0.001) identified 3 of 5 publishers with statistically significant bias — the kind of finding that has to survive scrutiny from a legal or compliance reviewer.
>
> My AI Safety Red-Team Evaluation project extends that pipeline: LLMs generate labels, a Stacking Classifier on 47 features generalizes them, and the whole loop runs at 850 samples/hour with Krippendorff's α = 0.81 and a 340× cost reduction versus human annotation. The pattern maps directly to S&P work — LLM-scored ratings actions, ESG classification, filings triage.
>
> Third, my Breast Cancer ML Classification project (99.12% accuracy, ROC-AUC 0.9987, ECE 0.0089 post-Platt calibration) shows I can also own the deterministic-supervised side end-to-end, with calibration and threshold policies suited to regulated decisions.
>
> Everything ships with MLflow, SHAP, and IEEE 2830-2025-aligned model cards. I have an MS in Applied Statistics (RIT, 2026), am authorized to work in the US, and available immediately for 2026 start.
>
> Thank you for the consideration.
>
> Derek Lankeaux

## Screener prep

- "How would you evaluate an LLM summarizing an earnings call?" → describe LLM-as-judge harness with pairwise-agreement α, and how to catch regressions with a champion/challenger set.
- "How do you quantify uncertainty in an LLM output?" → 95% HDI from PyMC hierarchical, plus calibration curves.
- Expect a SQL screen (S&P is heavy on SQL) — brush up on window functions and CTEs.

## Resume tweaks

- Reorder LLM Bias Detection above AI Safety on the resume — the S&P role weights document-NLP work more heavily.
- Add one line: "Structured for compliance review: model cards + audit trails aligned to ISO/IEC 23894."
