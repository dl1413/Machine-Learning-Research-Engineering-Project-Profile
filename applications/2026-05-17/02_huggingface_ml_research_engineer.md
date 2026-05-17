# 02 — Hugging Face · ML Research Engineer (Evaluation / Open Models)

**Location:** Remote (US)
**Tier:** B — Eval-native
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting:** LLM Bias Detection, Breast Cancer
**JD source:** huggingface.co/jobs — filter "Research Engineer" / "Evaluation"
**Resume version to send:** `resume_v_safety.pdf`

---

## Tailored Projects section

### AI Safety Red-Team Evaluation Framework — *Independent Research, Jan 2026*
- Built production eval harness processing **850 samples/hr** with circuit breakers, exponential backoff, and MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs (ROC-AUC 0.9923, precision 97.2%)
- Quantified judge disagreement with a PyMC Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family
- Designed 47 linguistic / semantic / structural features for jailbreak, refusal-evasion, and policy-violation signals
- Reproducible pipeline released alongside published technical report; IEEE 2830-2025 audit trail, SHAP explanations, $0.018/sample (340x cheaper than human review)

### LLM Ensemble Textbook Bias Detection — *Independent Research, 2025*
- 67,500 ratings, 2.5M tokens, full MLflow lineage on the same 3-LLM ensemble; alpha = 0.84, 92% pairwise correlation
- Published rubric + reproducible code; Friedman chi-squared = 42.73, p < 0.001 with Nemenyi post-hoc

### Clinical-Grade Breast Cancer Classification — *2025*
- 99.12% accuracy / ROC-AUC 0.9987; FastAPI <100ms p95 with MLflow model registry

---

## Cover letter

> Hi Hugging Face team,
>
> I'm interested in the Research Engineer / Evaluation role because the open-eval problem is the one I've been working on for the last six months. I built a 3-model LLM eval harness — GPT-4o, Claude-3.5, Llama-3.2 — that auto-grades 12,500 response pairs at **96.8% accuracy and 850 samples/hr**, with circuit breakers, async batching, and MLflow tracking baked in. Cost per sample landed at $0.018, a **340x reduction versus human review**. The interesting part is the stacking layer: a 47-feature meta-classifier that reconciles disagreement between the three judges and surfaces per-model blind spots via Bayesian hierarchical modeling (95% HDI per judge).
>
> Same ensemble, different domain: I ran 67,500 ratings through it for a textbook-bias study across 4,500 passages (alpha = 0.84, p < 0.001 on 3 of 5 publishers). Both projects are fully reproducible — code, priors, posteriors, and technical reports on GitHub (dl1413) — and that "release the eval harness, not just the result" mindset is what draws me to Hugging Face specifically.
>
> I'd love to contribute to open-eval tooling, leaderboards, or judge-quality research. Available remote, US East timezone, 2026 start after wrapping my MS in Applied Statistics at RIT.
>
> Best,
> Derek Lankeaux

---

## JD-keyword echo plan
- Phrase 1: ________________________
- Phrase 2: ________________________
- Phrase 3: ________________________
