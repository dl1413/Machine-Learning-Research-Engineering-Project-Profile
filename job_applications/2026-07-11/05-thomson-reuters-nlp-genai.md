# Thomson Reuters — Senior Applied Scientist, NLP / GenAI

**Location:** New York, NY
**Posting:** https://www.builtinnyc.com/job/senior-applied-scientist-nlp-genai/7469651
**Lead project:** LLM Ensemble Bias Detection + AI Safety Red-Team

---

## Cover Letter

Hi Thomson Reuters Applied Science team,

The Senior Applied Scientist, NLP / GenAI posting is a natural fit for what I've been building: multi-LLM evaluation systems with rigorous bias measurement, defensible statistics, and production engineering that survives real audit scrutiny — the exact shape of work Thomson Reuters needs for legal and news domains.

**Multi-LLM evaluation and bias measurement.** My **LLM Ensemble Textbook Bias Detection** framework (April 2026) ran GPT-4o, Claude-3.5, and Llama-3.2 across **4,500 passages / 67,500 ratings / 2.5M tokens**, achieving **Krippendorff's α = 0.84** and 92% pairwise correlation. A PyMC hierarchical model with partial pooling produced **publisher-level credible bias intervals (95% HDI, R-hat < 1.01)** and identified **3/5 publishers** with statistically significant bias (Friedman χ² = 42.73, p < 0.001). Spearman inter-publisher correlations (ρ up to 0.74) revealed structural editorial relationships; bootstrap CIs flagged 12.3% high-uncertainty passages for expert review. Translate "publisher" to "content source", "outlet", or "legal jurisdiction" and this is the framework Thomson Reuters can plug in against Westlaw / Reuters content flows.

**Safety and misuse evaluation.** My **AI Safety Red-Team Evaluation** shipped a dual-stage LLM ensemble + ML classifier over **12,500 response pairs / 6 harm categories** at **96.8% accuracy** (97.2% precision) and **340× lower cost per sample** than human labeling. MITRE ATLAS-aligned taxonomy, multi-turn escalation identified as top attack (31.8%), dual-filter defense measured to reduce harm 21.8% → 4.8%.

**Production and standards.** Circuit breakers, exponential backoff, MLflow, SHAP, FastAPI. All three reports are IEEE 2830-2025 / ISO/IEC 23894 / EU AI Act-aligned — increasingly required for content that touches regulated professional workflows.

Applied Statistics MS at RIT (expected 2026), NYC-flexible.

Thanks for the read,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)

---

## Attach

- Resume PDF
- `LLM_Bias_Detection_Publication.pdf`
- `AI_Safety_RedTeam_Evaluation_Publication.pdf`
