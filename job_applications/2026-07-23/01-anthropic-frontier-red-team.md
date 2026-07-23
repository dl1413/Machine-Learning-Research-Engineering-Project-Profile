# Anthropic — Research Engineer, Frontier Red Team (RSP Evaluations)

**Location:** Remote-friendly (SF / NYC hubs) · **Apply:** https://www.anthropic.com/careers
· **Fit:** ★★★★★

---

Dear Anthropic Frontier Red Team,

I'm applying for the Research Engineer role on the Frontier Red Team (RSP Evaluations).
Building automated systems that decide whether a frontier model is safe to release is
the exact problem I chose to spend the last year of my Applied Statistics MS on, and
the artifacts are already in the wild.

My **AI Safety Red-Team Evaluation Framework** is a dual-stage LLM-ensemble pipeline
(GPT-4o + Claude-3.5 + Llama-3.2 → Stacking Classifier) that scored 12,500 model
responses across 6 harm categories (dangerous info, hate, deception, privacy, illegal
activity, self-harm) with an 8-category MITRE ATLAS-aligned adversarial taxonomy. It
reached **96.8% accuracy, 97.2% precision, 96.1% recall, ROC-AUC 0.9923** at a
Krippendorff's α = 0.81 across annotators — audit-grade agreement at $0.018/sample
(340× cheaper than the $6.12 human baseline), running at 850 samples/hour with SHAP
explanations and MLflow-tracked audit trails. Multi-turn escalation surfaced as the
highest-risk vector at 31.8%; a dual-filter defense pattern I evaluated cut the
observed harm rate from 21.8% to 4.8%.

That project sits on top of two others that show the same pattern applied more
broadly: an **LLM Ensemble Bias Detection** system using PyMC hierarchical models
(R-hat < 1.01, 95% HDI, Friedman χ² = 42.73, p < 0.001) across 4,500 textbook
passages, and a **clinical-grade classifier** at 99.12% accuracy / 100% precision
with Platt-calibrated probabilities (ECE 0.0089) and threshold policies tuned for
100% sensitivity in screening. All three are IEEE 2830-2025 and ISO/IEC 23894:2025
aligned; reports are in `dl1413/Machine-Learning-Research-Engineering-Project-Profile`.

I'd love to bring this to the RSP evaluations you're building — happy to walk through
the taxonomy design and the ensemble/classifier stack in a first conversation.

Derek Lankeaux · [LinkedIn](https://linkedin.com/in/derek-lankeaux)
· [GitHub](https://github.com/dl1413)
