# 01 — Anthropic · Research Engineer, Frontier Red Team

| Field | Value |
|---|---|
| **Company** | Anthropic |
| **Role** | Research Engineer, Frontier Red Team |
| **Location** | Remote (US) / SF / NYC presence |
| **Source** | careers.anthropic.com (apply direct) |
| **JD link** | _VERIFY before submit:_ https://www.anthropic.com/careers (filter: Red Team / Safety) |
| **Lead project** | P1 — AI Safety Red-Team Evaluation |
| **Supporting** | P2 — LLM Bias Detection (multi-LLM eval rigor); P3 — Breast Cancer (calibration discipline) |
| **Resume version** | `resume_v3_safety.pdf` |
| **Cover letter** | Yes — see below |
| **Referral** | None — apply cold; ask LinkedIn 1st-degrees post-submit |

## JD keyword echo (fill in after reading JD)

- [ ] red-team / adversarial evaluation
- [ ] frontier model behavior
- [ ] [Anthropic-specific phrase 1]
- [ ] [Anthropic-specific phrase 2]
- [ ] [Anthropic-specific phrase 3]

## Resume bullets (top 4 — paste into "Selected Projects")

From `APPLICATION_SNIPPETS.md` → P1 → "AI Safety / Red-Team / Alignment":

- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

## Cover letter draft

Dear Anthropic Frontier Red Team,

My most relevant work for Anthropic's mission is an independent AI Safety
Red-Team Evaluation framework I built and published in early 2026. It
ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a
stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and
ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories.
The pipeline runs at 850 samples/hr for $0.018/sample — a 340x cost
reduction versus human annotation — while holding inter-rater reliability
at Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian
hierarchical model that produces 95% HDI risk intervals per judge, and
shipped the whole thing under IEEE 2830-2025 audit-trail requirements.

Two adjacent projects show I bring the same rigor outside safety: an
LLM-ensemble bias study that processed 67,500 ratings across 4,500
passages (Friedman χ² = 42.73, p < 0.001) with MCMC convergence
(R-hat < 1.01), and a clinical classifier at 99.12% accuracy / 100%
precision / ECE 0.0089 after Platt calibration — proof that I take
calibration and uncertainty seriously, not just point estimates.

I'd love to bring that combination of eval rigor and production throughput
to the Frontier Red Team's automated adversarial-evaluation work.

I'm based in / available for New York City and open to remote, targeting
a 2026 start once I wrap my MS in Applied Statistics at RIT. Portfolio,
code, and the three technical reports referenced above are on my GitHub
(dl1413). Happy to walk through any of them.

Best,
Derek Lankeaux
