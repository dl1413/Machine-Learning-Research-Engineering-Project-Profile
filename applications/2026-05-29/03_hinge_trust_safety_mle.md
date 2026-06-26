# 3. Hinge — Machine Learning Engineer, Trust & Safety

- **Location:** New York, NY
- **Submit:** https://www.builtinnyc.com/job/machine-learning-engineer-trust-safety/9303141
- **Role family:** Trust & Safety / Integrity ML
- **Lead project:** P2 — LLM Ensemble Bias Detection · **Supporting:** P1 (harm detection), P3 (rigor)

## JD keywords to echo (verbatim)
trust & safety · evaluating LLMs in real-world applications · baseline metrics · evaluation methodologies · applied scientist / data scientist · online harms

## Cover letter

Dear Hinge Trust & Safety Team,

I'm interested in this role because I've spent the last few months building
exactly the **evaluation methodologies** that **trust & safety** ML depends on.
My LLM-ensemble bias-detection study ran **67,500 ratings** through a
GPT-4o / Claude-3.5 / Llama-3.2 ensemble over 4,500 passages. The headline
finding — that 3 of 5 publishers showed statistically significant directional
bias (Friedman chi-squared = 42.73, p < 0.001) — only holds because the pipeline
was built to defend it: Krippendorff's alpha = 0.84 across raters, 92% pairwise
correlation, and **baseline metrics** validated against a PyMC partial-pooling
model with R-hat < 1.01.

That's **evaluating LLMs in real-world applications** with the reliability
scaffolding that integrity decisions require. My companion red-team project
extends it directly to **online harms**: a 3-model ensemble that auto-grades
12,500 response pairs across 6 harm categories at 96.8% accuracy and ROC-AUC
0.9923, at 850 samples/hr and $0.018/sample. I'd bring that combination — an
**applied scientist / data scientist** who can ship harm classifiers *and*
defend their metrics statistically — to Hinge's safety surface.

I'm based in New York City and open to remote, targeting a 2026 start once I
wrap my MS in Applied Statistics at RIT. Code and reports are on GitHub (dl1413).

Best,
Derek Lankeaux

## Resume — Projects section (lead order for this role)

**LLM Ensemble Textbook Bias Detection** — *Independent Research, Jan 2026*
- Designed multi-LLM bias-rating system over 4,500 passages / 2.5M tokens; surfaced significant publisher bias (p < 0.001) in 3/5 publishers
- Held inter-rater reliability at Krippendorff's alpha = 0.84 with 92% pairwise correlation across GPT-4o, Claude-3.5, Llama-3.2
- Built async API layer with circuit breakers + exponential backoff sustaining 67,500 ratings; PyMC hierarchical model (R-hat < 1.01) with 95% HDI per publisher

**AI Safety Red-Team Evaluation Framework** — *Independent Research, Jan 2026*
- 3-model harm-detection ensemble across 6 categories: 96.8% accuracy, ROC-AUC 0.9923, 850 samples/hr, 340x cheaper than human review

**Clinical-Grade Breast Cancer Classification** — *Independent Research, Jan 2026*
- 8-algorithm benchmark; 99.12% accuracy / 100% precision; FastAPI deployment < 100ms p95
