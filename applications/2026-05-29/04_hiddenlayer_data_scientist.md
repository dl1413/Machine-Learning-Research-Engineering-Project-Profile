# 4. HiddenLayer — Data Scientist (Security / LLM)

- **Location:** Remote (US)
- **Submit:** https://dynamitejobs.com/company/hiddenlayer/remote-job/data-scientist-2
- **Role family:** Security ML / LLM Eval
- **Lead project:** P1 — AI Safety Red-Team Evaluation · **Supporting:** P2, P3
- **Posting notes:** 3–4+ yrs production experience; hands-on LLM (prompting, context design, tool use, evaluation, fine-tuning); building/shipping LLM-powered systems for security products

## JD keywords to echo (verbatim)
LLM-powered systems · security products · prompting · context design · tool use · evaluation · production · ship models

## Cover letter

Dear HiddenLayer Team,

HiddenLayer sits at the intersection of **security products** and
**LLM-powered systems**, which is exactly where my most relevant project lives.
I built and published an AI Safety Red-Team **evaluation** framework that
ensembles GPT-4o, Claude-3.5, and Llama-3.2 as judges and trains a stacking
classifier on 47 harm-signal features — reaching 96.8% accuracy and ROC-AUC
0.9923 against a 12,500-pair benchmark across 6 harm categories. It's a
**production** pipeline: 850 samples/hr, circuit breakers, exponential backoff,
async batching, and MLflow tracking, at $0.018/sample (340x cheaper than human
review).

That work is hands-on across the full LLM surface you care about —
**prompting** and rubric design, **context design** for the judge ensemble,
**tool use** to orchestrate three model APIs, and rigorous **evaluation** with
Krippendorff's alpha = 0.81 and a PyMC Bayesian model for per-judge uncertainty.
I'm comfortable **shipping models** that have to hold up under adversarial
pressure, and I quantify their reliability rather than asserting it — which is
what a security product needs from its detection layer.

I'm open to remote (US-based) and targeting a 2026 start once I wrap my MS in
Applied Statistics at RIT. Code and three technical reports are on GitHub (dl1413).

Best,
Derek Lankeaux

## Resume — Projects section (lead order for this role)

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Built production eval harness: 850 samples/hr, circuit breakers, exponential backoff, async batching, MLflow run tracking
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into an XGBoost meta-classifier reaching 96.8% agreement with gold labels on 12,500 pairs; ROC-AUC 0.9923
- 47 engineered harm features (jailbreak, refusal-evasion, policy-violation); $0.018/sample, 340x cheaper than human review; PyMC 95% HDI per judge

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- 67,500 ratings / 2.5M tokens at production scale; Krippendorff's alpha = 0.84; significant bias at p < 0.001

**Clinical-Grade Breast Cancer Classification** — *Independent Research, Jan 2026*
- 8-algorithm benchmark; 99.12% accuracy / 100% precision / ROC-AUC 0.9987; FastAPI < 100ms p95
