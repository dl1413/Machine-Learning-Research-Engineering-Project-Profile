# 01 — Anthropic, Member of Technical Staff, Frontier Red Team

- **Location:** Remote (US) / SF / NYC presence
- **Source:** careers.anthropic.com (direct)
- **Role family:** AI Safety / Red-Team / Alignment
- **Lead project:** AI Safety Red-Team Evaluation Framework
- **Supporting projects:** LLM Bias Detection (multi-LLM eval rigor), Breast Cancer (productionization)
- **Resume version:** `resume_v3_safety.pdf`

## Cover letter (paste-ready)

Hi Anthropic team,

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

Two other projects round out the picture: a 67,500-rating multi-LLM bias
evaluation (Krippendorff's alpha = 0.84, Friedman chi-squared = 42.73,
p < 0.001) that taught me how to defend an eval result statistically, and a
clinical-grade ensemble classifier (99.12% accuracy, 100% precision) deployed
behind a <100ms p95 FastAPI service.

I'd love to bring that combination of eval rigor, Bayesian uncertainty
quantification, and production throughput to the Frontier Red Team. I'm
based in / available for NYC and open to remote, targeting a 2026 start once
I wrap my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux
LinkedIn: linkedin.com/in/derek-lankeaux | GitHub: github.com/dl1413

## Resume top-3 bullets (lead with these in Projects section)

1. Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
2. Cut human-eval cost 340x ($6.12 -> $0.018/sample) while maintaining Krippendorff's alpha = 0.81 across the three judges
3. Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals; shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability

## JD keyword targets (fill in after pulling JD)

- [ ] "red-team" / "red teaming" — appears in cover letter line 1
- [ ] "alignment" — add to resume summary
- [ ] "jailbreak" — appears in bullet 3
- [ ] "Constitutional AI" — tech stack line
- [ ] "evaluation" — appears throughout
