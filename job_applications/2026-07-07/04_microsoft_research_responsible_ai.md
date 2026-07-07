# 04 — Microsoft Research: Applied Research Scientist — Responsible AI / AI

**Location:** New York, NY (MSR NYC also considers hybrid)
**Careers portal:** https://www.microsoft.com/en-us/research/careers/open-positions/
**Verified:** MSR portal listed 132 open positions as of 2026-07-06. Filter for area = "Artificial Intelligence", location = New York. Confirm exact requisition before applying.

## What they want (MSR Applied Scientist patterns)

> Bridge research and product: publish novel methods AND ship them into a product team. Strong statistical or ML foundation, real-world evaluation experience, comfort with regulated / responsible-AI framing.

## Why this candidate fits

MSR NYC has a heavy responsible-AI and behavioral-science orientation (FATE group history, DeCoDeX, Bing team). Derek's IEEE 2830-2025 / ISO 23894 / EU AI Act alignment across all 3 projects is the exact vocabulary MSR RAI leadership uses. The Bayesian hierarchical modeling (PyMC + R-hat + HDI) is the statistical dialect MSR-NYC's causal/experimentation research is written in.

## Project → Requirement mapping

| MSR requirement | Anchor project | Evidence |
|---|---|---|
| Publishable research artifacts | All 3 (v2.0, v4.0, v4.0 tech reports) | 3 publication-grade reports |
| Novel evaluation methodology | AI Safety Red-Team | Dual-stage LLM+classifier architecture is a genuine contribution |
| Statistical rigor | LLM Bias Detection | Friedman χ², PyMC hierarchical, 95% HDI, R-hat, Bonferroni/FDR |
| Real-world deployment story | Breast Cancer ML | 8-algorithm benchmark, calibration, FastAPI <100ms p95 |
| Responsible AI standards fluency | All 3 | IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act artifacts |

## Cover letter

> Dear Microsoft Research team,
>
> I am applying for the Applied Research Scientist position at MSR NYC because the group's history — combining rigorous statistical thinking with real-product responsible-AI work — is where I want the next stage of my research to live.
>
> My AI Safety Red-Team Evaluation project developed a two-stage evaluation framework: an ensemble of frontier LLMs (GPT-4o, Claude-3.5, Llama-3.2) generates labels, a Stacking Classifier on 47 engineered features learns to predict harm across 12,500 response pairs and 6 categories. Results: 96.8% accuracy, Krippendorff's α = 0.81, and a 340× cost reduction versus human annotation. The MITRE ATLAS-aligned adversarial taxonomy identifies multi-turn escalation as the top attack vector (31.8%), and the dual-filter defense analysis quantifies mitigation at 78% harm reduction. Structured throughout to IEEE 2830-2025.
>
> My LLM Ensemble Bias Detection paper (v4.0) is closer to MSR-NYC's document-analysis and causal-adjacent work: 67,500 ratings, PyMC hierarchical model with partial pooling (R-hat < 1.01), 95% HDI-based bias detection at the publisher level, Friedman χ² = 42.73 (p < 0.001). Bootstrap CIs flag 12.3% of passages as high-uncertainty for expert review — a triage mechanism that generalizes beyond textbooks.
>
> My Breast Cancer ML Classification project (99.12% acc, 100% precision, ECE 0.0089 post-Platt calibration) shows I can also carry a full modeling pipeline from Optuna TPE hyperparameter search (5× more sample-efficient than grid) through SHAP-based fairness auditing. Fairness auditing framed to IEEE 2830-2025.
>
> All 3 projects are already written up as publication-grade technical reports. I hold an MS in Applied Statistics from RIT (2026), am authorized to work in the US, and would be excited to talk about how the Red-Team framework could seed a new evaluation methodology at MSR.
>
> Thank you for the consideration.
>
> Derek Lankeaux · linkedin.com/in/derek-lankeaux · github.com/dl1413

## Screener prep

- MSR loves depth-of-methods — expect to defend the PyMC prior choice, the α threshold interpretation, the Platt vs isotonic tradeoff.
- Prepare a 15-minute "chalk talk" summary of the Red-Team project. That's the standard MSR interview format.
- Publications matter — mention the 3 tech reports early and offer the PDFs.

## Resume tweaks

- Lead with the "3 publication-grade technical reports" line.
- Add explicit publication versions (v2.0, v4.0, v4.0) — MSR reads versioning as maturity.
- Elevate the standards-compliance line (IEEE 2830-2025 / ISO/IEC 23894 / EU AI Act) to the summary.
