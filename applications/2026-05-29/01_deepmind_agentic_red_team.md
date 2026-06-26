# 1. Google DeepMind — Senior Security Engineer, Agentic Red Team

- **Location:** New York / US (verify on posting; posted with US base band ~$166k–$244k + bonus + equity)
- **Submit:** https://job-boards.greenhouse.io/deepmind/jobs/7596438
- **Role family:** AI Safety / Red-Team
- **Lead project:** P1 — AI Safety Red-Team Evaluation · **Supporting:** P2 (LLM Bias), P3 (rigor)

## JD keywords to echo (verbatim)
red team · LLM architectures · agentic workflows · prompt injection · adversarial examples · exploits for GenAI models · evaluation suites · benchmarks · challenge datasets

## Cover letter

Dear DeepMind Hiring Team,

The work most relevant to the Agentic Red Team is an **AI Safety Red-Team
Evaluation** framework I built and published in January 2026. It ensembles
GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking
classifier on 47 harm-signal features, reaching **96.8% accuracy and ROC-AUC
0.9923** against a 12,500-pair benchmark across 6 harm categories. The pipeline
runs at 850 samples/hr for $0.018/sample — a **340x cost reduction** versus
human annotation — while holding inter-rater reliability at Krippendorff's
alpha = 0.81.

That project is essentially a **red team** and **evaluation suite** for **LLM**
outputs: I designed **challenge datasets** spanning jailbreak, refusal-evasion,
and policy-violation signals, and the 47-feature classifier exists to catch the
**adversarial examples** that slip past single-judge grading. I paired it with a
PyMC Bayesian hierarchical model that produces 95% HDI risk intervals per judge
— exactly the kind of uncertainty quantification you want before trusting a red
team's verdict on **agentic workflows** and **prompt injection** attempts.

I'm based in / available for New York City and open to remote, targeting a 2026
start once I wrap my MS in Applied Statistics at RIT. Code and the three
technical reports referenced above are on my GitHub (dl1413). The red-team eval
is probably the fastest way to see how I think about adversarial robustness for
DeepMind's agentic systems.

Best,
Derek Lankeaux

## Resume — Projects section (lead order for this role)

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Designed 47 linguistic/semantic/structural features capturing jailbreak, refusal-evasion, and policy-violation (prompt-injection) signals
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while holding Krippendorff's alpha = 0.81; shipped IEEE 2830-2025 audit pipeline with SHAP explainability
- Modeled multi-judge uncertainty via PyMC Bayesian hierarchy producing 95% HDI risk intervals per judge

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- Operated 3-LLM ensemble at production scale: 67,500 ratings, 2.5M tokens; surfaced significant bias (Friedman chi-squared = 42.73, p < 0.001) at Krippendorff's alpha = 0.84

**Clinical-Grade Breast Cancer Classification** — *Independent Research, Jan 2026*
- 8-algorithm benchmark; winning ensemble at 99.12% accuracy, 100% precision, ROC-AUC 0.9987, deployed behind FastAPI < 100ms p95
