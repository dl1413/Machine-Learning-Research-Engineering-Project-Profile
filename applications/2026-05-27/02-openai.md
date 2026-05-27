# 02 — OpenAI, Research Engineer, Safety Systems

- **Location:** New York City (hybrid)
- **Source:** openai.com/careers (direct)
- **Role family:** LLM Evaluation / Safety
- **Lead project:** AI Safety Red-Team Evaluation Framework
- **Supporting project:** LLM Bias Detection (eval scale, statistical rigor)
- **Resume version:** `resume_v3_safety.pdf`

## Cover letter (paste-ready)

Hi OpenAI team,

I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5,
Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and
850 samples/hr, with circuit breakers, async batching, and MLflow tracking
baked in. Cost per sample landed at $0.018, a 340x reduction versus human
review. The interesting part for Safety Systems is the stacking layer: a
47-feature meta-classifier that reconciles disagreement between the three
judges and surfaces per-model blind spots via a PyMC Bayesian hierarchical
model with 95% HDI risk intervals. The whole pipeline is documented under
IEEE 2830-2025 audit-trail requirements.

I followed it with a 67,500-rating multi-LLM bias study (4,500 passages,
2.5M tokens) that held Krippendorff's alpha = 0.84 and surfaced
publisher-level bias at Friedman chi-squared = 42.73, p < 0.001 — the kind
of result that only stands up when the eval scaffolding is built to defend
it.

I'd love to bring that habit — pair every eval claim with the uncertainty
machinery that makes it defensible — to the Safety Systems team in NYC.
Based in / available for New York City and open to remote, 2026 start once
my MS in Applied Statistics at RIT wraps.

Best,
Derek Lankeaux
LinkedIn: linkedin.com/in/derek-lankeaux | GitHub: github.com/dl1413

## Resume top-3 bullets

1. Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking
2. Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels on 12,500 pairs
3. Quantified judge disagreement with PyMC Bayesian hierarchical model (95% HDI), surfacing systematic per-model blind spots for downstream policy decisions

## JD keyword targets

- [ ] "evaluation" / "evals" / "eval harness"
- [ ] "LLM-as-judge"
- [ ] "model behavior"
- [ ] "safety"
- [ ] "production" / "scalable"
