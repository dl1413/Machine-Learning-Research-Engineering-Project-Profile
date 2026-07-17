# HiddenLayer — AI Red Teamer

**Location:** Remote • **Fit:** ⭐⭐⭐ Strong
**Apply:** https://job-boards.greenhouse.io/hiddenlayer/jobs/4599590007

**Why this fits:** HiddenLayer evaluates security of predictive AND generative AI models, identifies vulnerabilities, and simulates adversarial attacks — the exact shape of my AI Safety Red-Team project. I have both angles: the LLM-ensemble side for generative-model evaluation, and the stacking-classifier ML side for predictive-model auditing (SHAP explainability, calibration, threshold tuning).

**Lead project:** AI Safety Red-Team Evaluation
**Supporting:** Breast Cancer Classification (predictive-model auditing depth), LLM Bias Detection (multi-model eval infra)

---

## Cover Letter

Dear HiddenLayer Team,

The reason I'm applying is that HiddenLayer sits in the exact place I've been trying to reach: adversarial evaluation of BOTH predictive and generative AI systems, with security as the framing. Most red-team work I see is one or the other. Mine has been both, in parallel, this year.

On the generative side: I built an end-to-end LLM red-team framework — a dual-stage pipeline where a three-model ensemble (GPT-4o, Claude-3.5, Llama-3.2) labels harm across six categories, then a stacking classifier over 47 engineered features renders the final decision. On 12,500 AI response pairs it reached 96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923), and the ensemble held Krippendorff's α = 0.81 — reliable enough to replace human annotation at 340× lower cost. I structured attacks along MITRE ATLAS and quantified that multi-turn escalation carried the highest harm rate (31.8%); a two-layer defense dropped overall harm 21.8% → 4.8%.

On the predictive side: I benchmarked eight algorithms (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting) on a clinical classification task, reached 99.12% accuracy with 100% precision, and — the piece that matters for auditing predictive AI — used Platt calibration to reduce ECE from 0.0312 to 0.0089, tuned decision thresholds for context-specific policies (e.g., 100% sensitivity for screening), and shipped SHAP explanations plus IEEE 2830-2025-aligned model cards.

That combination — LLM ensemble evaluation of generative systems and calibrated, explainable auditing of predictive systems — is what I'd bring to HiddenLayer.

Best,
Derek Lankeaux

---

## Resume-Bullet Variant

- Delivered dual-stage LLM red-team framework (GPT-4o + Claude-3.5 + Llama-3.2 → Stacking Classifier) reaching 96.8% accuracy across 12,500 harm-labeled response pairs at 340× cost reduction versus human annotation
- Structured adversarial attack taxonomy along MITRE ATLAS; identified multi-turn escalation as highest-risk vector (31.8%) and validated dual-filter defense (78% harm reduction: 21.8% → 4.8%)
- Audited predictive ML systems end-to-end: 8-algorithm benchmark on clinical classification (99.12% acc, 100% precision), Platt calibration cut ECE 71.5% (0.0312 → 0.0089), SHAP + IEEE 2830-2025 model cards
- Production MLOps for both sides: circuit breakers, exponential backoff, MLflow experiment tracking, FastAPI <100ms p95 latency

---

## 60-Second Hook

"HiddenLayer is the rare AI-security shop that evaluates BOTH predictive and generative models. I've built both. On the generative side: three-LLM ensemble red-team pipeline, 96.8% accuracy across 12,500 responses, MITRE ATLAS taxonomy, dual-filter defense that cut harm from 22% to 5%. On the predictive side: eight-algorithm benchmark reaching 99.12% accuracy with SHAP explanations, calibrated probabilities, and IEEE 2830-2025 model cards. That symmetry is the reason I want to work at HiddenLayer specifically."
