# 03 — Hugging Face, ML Research Engineer, Evaluations

- **Location:** Remote (US)
- **Source:** apply.workable.com/huggingface (direct)
- **Role family:** LLM Eval / Research Engineer
- **Lead project:** LLM Ensemble Textbook Bias Detection (eval-tooling fit)
- **Supporting project:** AI Safety Red-Team Evaluation (harm-focused eval)
- **Resume version:** `resume_v3_eval.pdf`

## Cover letter (paste-ready)

Hi Hugging Face team,

I'm interested in the Evaluations role because I've spent the last few
months building exactly the kind of multi-LLM evaluation infrastructure
your `evaluate` and `lighteval` libraries abstract. My textbook-bias study
ran 67,500 ratings through a GPT-4o / Claude-3.5 / Llama-3.2 ensemble over
4,500 passages (2.5M tokens), with circuit-breakered async API integration
and MLflow lineage end-to-end. The eval rubric held Krippendorff's
alpha = 0.84 and 92% pairwise inter-LLM correlation, and surfaced
publisher-level bias at p < 0.001 — backed by a PyMC partial-pooling
hierarchical model with R-hat < 1.01 producing 95% HDI credible intervals
per publisher.

The companion red-team project is the harm-focused twin: a 3-model
stacking ensemble (47 features, 96.8% accuracy, ROC-AUC 0.9923) that
auto-grades 12,500 response pairs at 850 samples/hr for $0.018/sample,
340x cheaper than human annotation. Both projects are published as
research-grade reports and reproducible pipelines on my GitHub.

I'd love to help turn that kind of eval scaffolding into a product. Open
to remote and available for a 2026 start.

Best,
Derek Lankeaux
LinkedIn: linkedin.com/in/derek-lankeaux | GitHub: github.com/dl1413

## Resume top-3 bullets

1. Operated 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage
2. Engineered prompt templates and rubric for bias scoring; validated rubric stability via Krippendorff's alpha = 0.84 and 92% pairwise inter-LLM correlation
3. Published reproducible technical report with methodology, priors, sensitivity analysis, and full posterior visualizations

## JD keyword targets

- [ ] "evaluation" / "benchmarks"
- [ ] "open source"
- [ ] "Transformers" / "datasets"
- [ ] "reproducibility"
- [ ] "leaderboard"
