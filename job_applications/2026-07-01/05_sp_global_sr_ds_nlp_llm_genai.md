# Cover Letter — S&P Global, Sr Data Scientist — NLP, LLM and GenAI

**Job link:** https://www.efinancialcareers.com/jobs-United_States-New_York-Sr_Data_Scientist-_NLP_LLM_and_GenAI.id22809240
**Location:** New York, NY
**Salary range:** $85,000 – $150,000

---

Dear S&P Global Hiring Team,

I'm applying for the Sr Data Scientist — NLP, LLM and GenAI role. The scope you describe — custom ML/NLP/LLM models across batch and streaming pipelines, RAG, fine-tuning, prompt engineering, and applied research on Gen AI — is the shape of the three research projects I've published this year.

**Applied LLM research with a production pipeline.** My AI Safety Red-Team Evaluation framework runs a dual-stage LLM ensemble (GPT-4o + Claude-3.5 + Llama-3.2) into a Stacking Classifier over 47 engineered features. Across 12,500 responses and six harm categories it delivers 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923, Krippendorff's α = 0.81, and 340× cost reduction ($0.018 vs. $6.12 per sample). The pipeline runs at 850 samples/hour behind circuit breakers, exponential backoff, and MLflow tracking — the kind of infrastructure S&P needs to move NLP work from prototype to product.

**Statistical rigor for financial-grade decisions.** For LLM Ensemble Bias Detection I designed a PyMC hierarchical model over 67,500 ratings across 4,500 passages (2.5M tokens): partial pooling, R-hat < 1.01, Friedman χ² = 42.73 (p < 0.001), 95% HDIs, Bonferroni/FDR correction, and bootstrap CIs that flag high-uncertainty items for human review. Publisher-level Spearman correlation matrices (ρ up to 0.74) surfaced structural editorial relationships that a naive averaging pipeline would have missed. The same techniques carry directly to source-level bias, drift, and reliability analysis on financial content.

**Predictive modeling with calibration and explainability.** My clinical breast-cancer classifier: 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987, Platt-calibrated (ECE 0.0089), SHAP-explained, deployed via FastAPI/MLflow at < 100ms p95. Model cards are aligned with IEEE 2830-2025 and ISO/IEC 23894:2025 — increasingly the vocabulary regulated industries expect.

I'm completing an MS in Applied Statistics at RIT in 2026 (Bayesian methods focus) and I'm authorized to work in the US. Based in the NYC area. Portfolio, technical reports, and code: github.com/dl1413.

Best regards,
Derek Lankeaux
dlankeaux12@gmail.com | linkedin.com/in/derek-lankeaux
