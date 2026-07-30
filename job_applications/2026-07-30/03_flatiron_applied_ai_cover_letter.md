# Cover Letter — Flatiron Health, Senior Applied AI Data Scientist

**Role:** Senior Applied AI Data Scientist
**Location:** New York, NY (hybrid)
**Apply:** https://www.builtinnyc.com/job/senior-applied-ai-data-scientist/8615163
**Lead project:** Clinical-Grade Breast Cancer ML Classification (+ AI Safety Red-Team as GenAI evidence)

*Note: Senior title — the ML+GenAI-on-oncology-text framing lines up with my portfolio; worth submitting even if experience skews junior.*

---

Dear Flatiron Health team,

Flatiron's mission — extracting clinically relevant information from unstructured medical notes to build de-identified oncology research datasets — sits at the exact intersection I've been working at: high-stakes clinical ML on one side, LLM-ensemble evaluation on the other. My three published projects cover both halves.

**The clinical-ML half.** I built a breast-cancer classification system as an independent research project (April 2026): 8-algorithm ensemble benchmark (Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting) with nested cross-validation, VIF-based multicollinearity pruning, SMOTE class-balancing, and RFE feature selection. Best model landed at **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** — comfortably above the 90–95% range cited for human expert reads. I applied Platt calibration to bring ECE from 0.0312 to 0.0089 and tuned thresholds for context-specific decision policies (100% sensitivity at 0.31 for mass screening). SHAP explanations run per prediction, and the FastAPI service holds under 100ms p95. Everything ships under **IEEE 2830-2025** transparency and **ISO/IEC 23894:2025** AI-risk-management alignment — the artifacts a life-sciences ML team increasingly needs.

**The GenAI-for-unstructured-text half.** I've also shipped an LLM-ensemble evaluation framework — GPT-4o / Claude-3.5 / Llama-3.2 — that processes 12,500 unstructured response pairs at **96.8% accuracy, Krippendorff's α = 0.81, at $0.018/sample (340× cheaper than human annotation)**. And a multi-LLM bias-detection study over 2.5M tokens of long-form text where a PyMC hierarchical model (R-hat < 1.01, 95% HDI) surfaced credible publisher-level signal. That is the same "grade a large corpus of long text reliably and defend the result statistically" workflow you'd want against oncology notes.

Together: I can build the classifier, quantify the uncertainty, deploy the FastAPI service, and grade LLM extractions from unstructured notes with the reliability scaffolding Flatiron's research datasets need. I'm completing an MS in Applied Statistics at RIT, targeting a 2026 start, available for NYC hybrid, US-authorized.

Portfolio and full technical reports: github.com/dl1413.

Best,
Derek Lankeaux
