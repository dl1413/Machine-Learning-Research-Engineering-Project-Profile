# Application 2 — OpenAI · Member of Technical Staff, Red Team

| Field | Value |
|---|---|
| Date | 2026-05-25 (Mon) |
| Company | OpenAI |
| Role | Member of Technical Staff, Red Team |
| Role family | AI Safety / Red-Team |
| Location | NYC (hybrid) |
| Source | openai.com/careers |
| Lead project | Project 1 — AI Safety Red-Team Evaluation Framework |
| Supporting | Project 2 — LLM Bias Detection (multi-LLM eval at scale) |
| Resume version | resume_v3_safety.pdf |
| Cover letter | Yes |
| Referral | No |

---

## Tailored resume bullets

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- Stacked GPT-4o, Claude-3.5, and Llama-3.2 judges into a meta-classifier reaching 96.8% agreement with gold human labels on 12,500 jailbreak / policy-violation pairs
- Cut auto-grading cost to $0.018/sample (340x cheaper than human annotation) while holding Krippendorff's alpha = 0.81
- Built production eval harness at 850 samples/hr with circuit breakers, async batching, exponential backoff, and MLflow lineage
- Quantified per-judge blind spots via PyMC hierarchical model (R-hat < 1.01, 95% HDI per harm category)

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- Operated the same 3-LLM ensemble at 67,500-rating scale across 2.5M tokens, alpha = 0.84, 92% pairwise correlation
- Surfaced publisher-level bias at Friedman chi-squared = 42.73, p < 0.001 with post-hoc Nemenyi localization

## Cover letter

Dear OpenAI Hiring Team,

I'm applying for the Red Team MTS role because the eval work I've spent the
last year building maps directly onto what your team ships. In January 2026 I
published an AI Safety Red-Team Evaluation framework that ensembles GPT-4o,
Claude-3.5, and Llama-3.2 as judges and trains a 47-feature stacking
classifier across 6 harm categories — **96.8% accuracy, ROC-AUC 0.9923**, on
a 12,500-pair benchmark. Throughput is **850 samples/hr at $0.018/sample, a
340x reduction** versus human annotation, with Krippendorff's alpha = 0.81 to
defend the labels. A PyMC Bayesian hierarchy produces 95% HDI risk intervals
per judge, and the whole pipeline runs under IEEE 2830-2025 audit standards.

The same stack scaled to 67,500 ratings across 2.5M tokens of textbook content
for a bias-detection study (alpha = 0.84, Friedman p < 0.001 on publisher
effects) — i.e., the eval infra holds up at production volume, not just on
toy benchmarks.

I'm wrapping my MS in Applied Statistics at RIT this year, based in NYC, and
ready to start in 2026. Excited to talk about how this fits the Red Team's
near-term agenda.

Best,
Derek Lankeaux

## JD keyword echoes

- "red team" / "red-teaming"
- "jailbreak"
- "harm" / "harmful outputs"
- "evaluation"
- "model behavior"

## Quality checklist

- [x] Lead project at top of Projects section
- [x] Metric hook (340x + 96.8%)
- [x] 3+ JD phrases verbatim
- [x] NYC availability stated
- [x] Salary: "open, targeting market for NYC"
- [x] Work auth: US authorized
