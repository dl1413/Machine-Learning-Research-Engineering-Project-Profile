# Cover Letter — Capital One, Principal Associate Data Scientist (LLM Customization)

Dear AI Foundations Team,

I'm applying for the Principal Associate, Data Scientist role on the LLM Customization team. What drew me is the scope you describe — that the team touches every stage of the LLM lifecycle, from research through production — because that end-to-end shape is the one my three main projects were designed around.

The closest analogue is my LLM Ensemble Bias Detection system. Across 4,500 passages and 67,500 ratings I orchestrated GPT-4o, Claude-3.5-Sonnet, and Llama-3.2 through a LangChain-mediated pipeline with circuit breakers, exponential backoff, and MLflow experiment tracking — 2.5M tokens at production scale — and pairwise correlation across models held at 92%. The evaluation layer that sits on top is where the customization posture matters: a PyMC hierarchical model with partial pooling (R-hat < 1.01) reports 95% HDI intervals at publisher and topic levels, flags 12.3% high-uncertainty items for expert review rather than auto-decide, and defends the aggregate finding with a Friedman χ² = 42.73 (p < 0.001). For a customization team choosing which fine-tuned checkpoint to ship, that's the shape of eval that turns "the numbers look better" into "we can defend this to model risk."

I've paired that with an AI Safety Red-Team framework that combines LLM-as-judge annotation and a downstream Stacking Classifier over 47 engineered features — 12,500 response pairs across 6 harm categories at 96.8% accuracy, Krippendorff's α = 0.81, and $0.018/sample (a 340× cost reduction over $6.12 human annotation). That business result is the one I'd want to plug into a customization workflow: it makes running the eval a per-week, not per-quarter, decision, and it's the pattern that scales from safety evals to any structured quality signal a customization team needs at high frequency.

Because Capital One is a regulated environment, the governance side of the resume matters too: SHAP audit trails, model cards, and IEEE 2830-2025 / ISO/IEC 23894:2025 / EU AI Act alignment are built into every project, not layered on. My Applied Statistics MS gives me the Bayesian and causal-inference tools the model-risk conversation eventually needs.

I'd love to talk about how the LLM Customization team currently structures its evals, which pieces are LLM-judged vs human-annotated today, and where a hierarchical Bayesian layer could tighten ship / no-ship decisions.

Warmly,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
