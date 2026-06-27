# Cover Letter — Anthropic, Research Engineer, Model Evaluations

Dear Anthropic Hiring Team,

I'm applying for the Research Engineer, Model Evaluations role on the Frontier
Red Team. The reason I lead with this role is that I've spent the last six
months building exactly the kind of evaluation infrastructure you describe —
and I'd rather build it for Claude than for anything else.

My most relevant work is an independent AI Safety Red-Team Evaluation
framework I published in January 2026. It ensembles GPT-4o, Claude-3.5, and
Llama-3.2 as red-team judges and trains a stacking classifier on 47 harm-signal
features, reaching **96.8% accuracy and ROC-AUC 0.9923** against a 12,500-pair
benchmark across 6 harm categories. The pipeline runs at **850 samples/hr for
$0.018/sample — a 340x cost reduction versus human annotation** — while
holding inter-rater reliability at Krippendorff's alpha = 0.81. I paired it
with a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals
per judge, and shipped the whole thing under IEEE 2830-2025 audit-trail
requirements with circuit breakers, exponential backoff, and full MLflow run
lineage so the eval can be re-run end-to-end from any commit.

Two adjacent projects round out the eval-engineering signal: an LLM-ensemble
bias-detection study (67,500 ratings over 4,500 textbook passages, alpha =
0.84, Friedman chi-squared = 42.73, p < 0.001 across publishers), and a
clinical-grade classifier (99.12% accuracy, 100% precision, ROC-AUC 0.9987)
that taught me how to ship a model when wrong answers carry a real cost.

What I'd want to bring to Claude's evals: the Bayesian-first habit of reporting
uncertainty alongside point estimates, an instinct for building eval harnesses
that don't lie when the model under test gets cleverer, and a willingness to
sit with researchers and chase down per-capability blind spots until the
numbers actually mean something.

I'm based in / available for New York City, targeting a 2026 start once I wrap
my MS in Applied Statistics at RIT. Code, reports, and the three technical
write-ups referenced above are on my GitHub (dl1413). Happy to walk through
the red-team eval as the fastest way to see how I think about Anthropic's
evaluation work.

Best,
Derek Lankeaux
