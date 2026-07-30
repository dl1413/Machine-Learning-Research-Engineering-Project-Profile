# Cover Letter — Hugging Face, Multimodal Research Engineer

**Role:** Multimodal Research Engineer (US Remote)
**Location:** US Remote
**Apply:** https://startup.jobs/multimodal-research-engineer-us-remote-hugging-face-5298815
**Lead project:** AI Safety Red-Team Evaluation Framework (eval pipelines)

---

Dear Hugging Face team,

The line in the JD that resonated most is *"design, build and maintain evaluation pipelines to answer research questions"* — that's been my primary independent-research thread for the last twelve months, and the reason I'd want to do it inside Hugging Face specifically.

**The eval infrastructure.** I built an AI Safety Red-Team Evaluation framework (published April 2026) that ensembles GPT-4o, Claude-3.5, and Llama-3.2 as judges and trains a stacking meta-classifier on 47 linguistic / semantic / structural harm-signal features. Against a **12,500 response-pair benchmark across 6 harm categories** it hits **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**, with inter-rater reliability at **Krippendorff's α = 0.81**. The pipeline runs at **850 samples/hour** with circuit breakers, async batching, exponential backoff, and MLflow-tracked runs — the boring but load-bearing scaffolding that lets an eval question actually get answered.

**Two things that would carry over immediately to multimodal work:**

1. *Judge disagreement is a signal.* I fit a PyMC Bayesian hierarchical model over the three judges (R-hat < 1.01, 95% HDI) to quantify per-model-family blind spots. Same instrument scales to the DPO-flavored preference-learning research your team runs — you want to know when annotators/judges systematically diverge, not just their mean.

2. *Adversarial taxonomy as a first-class object.* An 8-category MITRE ATLAS-aligned taxonomy (with multi-turn escalation flagged as the top risk vector at 31.8%) was what let a dual-filter defense drop measured harm rate 21.8% → 4.8%. That structured red-team library maps directly to safety evals for new modalities as inputs and outputs expand.

**Supporting projects** — a multi-LLM bias study (67,500 ratings, α = 0.84, Friedman χ² = 42.73, p < 0.001) and a clinical-grade ensemble classifier (99.12% accuracy, ROC-AUC 0.9987) — round out the toolkit with the Bayesian rigor and ensemble-modeling range you'd want on the training-recipe side.

I'm completing an MS in Applied Statistics at RIT (Bayesian methods, experimental design, deep learning), targeting a 2026 start, US remote friendly, US work-authorized. Full portfolio and technical reports: github.com/dl1413.

Best,
Derek Lankeaux
