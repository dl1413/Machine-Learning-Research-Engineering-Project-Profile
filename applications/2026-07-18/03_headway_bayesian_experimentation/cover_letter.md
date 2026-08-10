# Cover Letter — Headway, Senior Staff Data Scientist (Bayesian Experimentation & Causal Inference)

Dear Headway Data Science Team,

I'm applying for the Senior Staff Data Scientist role owning Bayesian experimentation and causal inference. The framing on your JD — that Headway wants **canonical approaches and guardrails** for when and how to use Bayesian methods — is the shape of work I've been practicing across three research projects.

The closest analogue is my LLM Ensemble Textbook Bias Detection project. Across 4,500 passages and 67,500 ratings I built a PyMC hierarchical model with partial pooling at the publisher level, converged with R-hat < 1.01 and full ESS diagnostics, and reported 95% HDIs at the publisher and topic layers — flagging 3/5 publishers with credibly non-zero bias and 12.3% of passages as high-uncertainty for expert review. I paired the Bayesian result with a Friedman χ² = 42.73, p < 0.001 as a non-parametric sanity check, and a Spearman correlation matrix showing structural editorial relationships (ρ up to 0.74). That "run the Bayesian model, corroborate it, and only decide when both agree" pattern is my starting point for a canonical experimentation guardrail: teams get pooled effect estimates with honest uncertainty, and reviewers can't wave off surprising results with distributional objections.

The AI Safety Red-Team project extends the same posture into multi-model risk: a Bayesian hierarchical model over three LLMs (GPT-4o, Claude-3.5, Llama-3.2) with 95% HDI intervals per harm category, sitting alongside Krippendorff's α = 0.81 as a reliability floor. It's an example of using Bayesian methods where frequentist framings undersell the joint uncertainty across raters and categories — I'd expect similar patterns to show up in a marketplace like Headway's (provider effects, patient effects, geo effects) where partial pooling is the right default.

For breadth: my breast-cancer classifier project (99.12% accuracy, 100% precision) uses Platt calibration (ECE 0.0089) and threshold policy tuning for context-dependent decisions — the same statistical toolkit that experimentation teams use to define ship criteria and cost-of-error tradeoffs. And the Applied Statistics MS at RIT (Bayesian Methods / Experimental Design / Causal Inference specialization) is the theoretical spine behind all three.

At Headway I'd want to talk about your current experimentation stack, where teams already reach for Bayesian methods versus where they'd benefit from a canonical guardrail, and how the analytics org communicates uncertainty to clinicians and product without over-claiming.

Warmly,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
