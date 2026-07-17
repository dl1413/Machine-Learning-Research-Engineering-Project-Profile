# 10a Labs — AI Red Teamer (Entry Level)

**Location:** Remote (US-based) • **Comp:** $60K–$70K • **Fit:** ⭐⭐⭐ Strong
**Apply:** https://job-boards.greenhouse.io/10alabs/jobs/4002004009

**Why this fits:** 10a Labs does adversarial red-teaming and model evaluations for frontier AI labs and Fortune 10 companies. My AI Safety Red-Team project is the single closest 1:1 match in my portfolio: dual-stage LLM ensemble + ML classification, 6 harm categories, MITRE ATLAS-aligned taxonomy, 340× cost reduction over human annotation. Entry-level requirement, so YOE is not a blocker.

**Lead project:** AI Safety Red-Team Evaluation
**Supporting:** LLM Ensemble Bias Detection (multi-LLM eval infra), Breast Cancer (ML rigor)

---

## Cover Letter

Dear 10a Labs Team,

I'm applying for the AI Red Teamer (Entry Level) role because the work you do — adversarial evaluation of frontier AI systems for the labs that build them — is the exact intersection I've spent the last year building toward.

My most recent research project is an end-to-end AI safety red-team evaluation framework. I built a dual-stage pipeline: a three-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) generates first-pass harm annotations across six categories, then a stacking classifier over 47 engineered linguistic, semantic, and structural features produces the final label. Across 12,500 AI response pairs the system reached 96.8% accuracy, 97.2% precision, and 96.1% recall (ROC-AUC 0.9923), and Krippendorff's α = 0.81 confirmed the LLM ensemble was reliable enough to substitute for human raters at $0.018/sample — a 340× cost reduction versus a $6.12 human baseline. I organized the attack taxonomy along MITRE ATLAS categories and found that multi-turn escalation carried the highest per-attempt harm rate (31.8%), while a dual-filter defense reduced overall harm from 21.8% to 4.8%.

I care about doing this work rigorously: Bayesian hierarchical modeling for cross-model risk, SHAP for auditability, MLflow tracking so every result is reproducible, and IEEE 2830-2025 / ISO/IEC 23894:2025 alignment because red-team work is only useful if the findings hold up.

I'd bring that same rigor to 10a Labs' engagements. Happy to walk through the framework in more detail.

Best,
Derek Lankeaux

---

## Resume-Bullet Variant (paste into resume for this application)

- Built dual-stage LLM red-team framework (GPT-4o, Claude-3.5, Llama-3.2 ensemble → Stacking Classifier over 47 features) achieving 96.8% accuracy across 12,500 responses and 6 harm categories, with Krippendorff's α = 0.81 confirming ensemble reliability
- Reduced safety-eval cost 340× ($0.018/sample vs $6.12 human baseline) at 850 samples/hr production throughput, preserving audit-grade reliability
- Aligned taxonomy to MITRE ATLAS; identified multi-turn escalation as highest-risk vector (31.8% harm rate) and validated a dual-filter defense (21.8% → 4.8% harm, 78% reduction)
- Quantified cross-model vulnerability with Bayesian hierarchical modeling (95% HDI); shipped SHAP-based audit trails per IEEE 2830-2025 / ISO/IEC 23894:2025

---

## 60-Second Hook (phone screen)

"I built an AI red-team framework end-to-end this year — three-model LLM ensemble annotating harm across six categories, stacking classifier over engineered features, 96.8% accuracy on 12,500 responses at 1/340th the cost of human annotation. I organized the attack taxonomy along MITRE ATLAS, so I've thought carefully about multi-turn escalation, jailbreak vectors, and defense stackups. That's why 10a Labs' work — adversarial evaluation for the labs actually building frontier models — is the role I want most."
