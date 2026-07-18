# Cover Letter — Uber, Senior Applied Scientist (AI Red Teaming & Model Risk)

Dear Uber Applied Science Hiring Team,

I'm applying for the Senior Applied Scientist role on AI Red Teaming & Model Risk in New York. The phrase in your posting that hooked me was "reusable evaluation pipelines to support continuous red teaming" — that's the thesis behind the framework I've been building.

My AI Safety Red-Team Evaluation Framework is a two-stage pipeline: a GPT-4o / Claude-3.5 / Llama-3.2 ensemble annotates model outputs across six harm categories, and a Stacking Classifier over 47 engineered features (linguistic, semantic, structural) scores them. It processes 850 samples/hour at 96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923) with $0.018/sample cost — a 340× reduction versus the $6.12 human-annotation baseline. Inter-rater reliability across the three LLMs is Krippendorff's α = 0.81, at the audit-grade threshold. Just as important for a Trust & Safety context: the framework quantifies **defense effectiveness** — a dual-filter defense in the study reduced harm rate from 21.8% → 4.8% (78% reduction), measured against a MITRE ATLAS-aligned 8-vector attack taxonomy where multi-turn escalation surfaced as the highest-risk pathway at 31.8% success.

Because model-risk work lives or dies by governance artifacts, I built the framework to produce them by default: PyMC hierarchical risk models with 95% HDI intervals, SHAP explainability, model cards, and IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act-aligned documentation. Those aren't polish — they're what turns a red-team writeup into something Legal and Policy can sign off on.

Two adjacent projects show I can move the pattern across domains without rebuilding: the same 3-model ensemble structure powers a textbook-bias detector (67,500 ratings, α = 0.84, Friedman χ² = 42.73, p < 0.001, PyMC partial pooling with R-hat < 1.01), and a clinical-grade breast-cancer classifier (99.12% accuracy, 100% precision, Platt calibration to ECE 0.0089, FastAPI < 100 ms p95) grounds me in production ML hygiene end-to-end.

At Uber the interesting extension would be adapting the framework for the specific surfaces where GenAI ships — customer-facing agent flows, driver-support LLM responses, internal copilots — each with its own harm taxonomy and defense stack. I'd want to talk about how continuous evals compose across those surfaces without duplicating infrastructure.

I'm based in the US, authorized to work, and comfortable in a NY hybrid cadence.

Warmly,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
