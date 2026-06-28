# Patronus AI — Research Scientist

**Posting:** https://www.glassdoor.com/job-listing/research-scientist-patronus-ai-inc-JV_KO0,18_KE19,34.htm
**Location:** Remote
**Why fit:** Patronus is automated LLM evaluation — my entire portfolio is about doing exactly that with statistical rigor (Krippendorff's α, Bayesian HDI, multi-model ensembles).

---

Dear Patronus AI team,

I'm applying for the Research Scientist role. LLM evaluation — the robustness, reliability, and bias surface area you focus on — is what I've spent the past year building, and three published projects directly extend the Patronus problem space.

**AI Safety Red-Team Evaluation** is the closest match. I designed a dual-stage framework: a GPT-4o / Claude-3.5 / Llama-3.2 ensemble annotates 12,500 adversarial response pairs across 6 harm categories, then a Stacking Classifier on 47 linguistic / semantic / structural features hits 96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923) — at $0.018 per sample versus $6.12 for human annotation (340× cost reduction) while preserving Krippendorff's α = 0.81. The taxonomy is MITRE ATLAS-aligned, with multi-turn escalation flagged as the highest-risk vector (31.8%), and the defense analysis shows dual-filter reducing harm rate 21.8% → 4.8%. This is exactly the kind of judge-plus-classifier architecture Patronus's API ships.

**LLM Ensemble Textbook Bias Detection** extends the same pattern to a subtler signal: 67,500 ratings, Bayesian hierarchical model in PyMC with partial pooling, MCMC convergence at R-hat < 1.01, and Friedman χ² = 42.73 (p < 0.001) for publisher-level credible bias — including a cross-topic heatmap surfacing 1.36-point publisher polarization on social issues. The 92% pairwise correlation across frontier LLMs is a useful empirical baseline for any LLM-as-judge product.

**Breast Cancer Classification** rounds out the engineering hygiene — calibration (ECE 0.0089 after Platt scaling), threshold policies, FastAPI deployment under 100ms p95.

I'd be excited to talk about extending these patterns to Patronus's evaluation primitives. Portfolio: https://dl1413.github.io/LLM-Portfolio/

Best,
Derek Lankeaux
