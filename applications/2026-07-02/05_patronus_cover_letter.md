# Cover Letter - Patronus AI, Research Scientist: AI Evaluation & Alignment

**Location:** Remote
**Lead project:** AI Safety Red-Team Evaluation Framework
**JD link:** https://jobright.ai/jobs/info/695edf200badca5763af5937

---

Dear Patronus AI team,

I'm interested in Patronus because I've spent the last few months
building - independently - the kind of multi-LLM evaluation
infrastructure your platform productizes. The Research Scientist role,
which sits on redteaming, automated evaluation, and alignment
research, matches the trajectory I want to be on.

**The most relevant project.** An AI Safety Red-Team Evaluation
framework that ensembles GPT-4o, Claude-3.5, and Llama-3.2 as
red-team judges and trains a stacking classifier on 47 harm-signal
features (linguistic, semantic, structural). Results on a
12,500-pair benchmark across 6 harm categories:

- **96.8% accuracy, 97.2% precision, ROC-AUC 0.9923**
- **Krippendorff's alpha = 0.81** across the three judges
- **340x cost reduction** versus human annotation ($0.018/sample vs $6.12)
- **850 samples/hr** at production scale
- **Bayesian hierarchical uncertainty modeling** (PyMC, 95% HDI) over
  the judge ensemble, so we can quantify *judge disagreement* on top
  of accuracy - crucial when you want to trust an automated eval as a
  guardrail
- **Multi-turn escalation** identified as the highest-risk attack
  vector (31.8% of adversarial hits) via a MITRE ATLAS-aligned taxonomy

**A second project extends the same pattern.** An LLM-ensemble bias
detection system running the same 3-LLM stack across 4,500 passages
and 2.5M tokens (67,500 ratings) with a PyMC hierarchical model
(R-hat < 1.01) that surfaces statistically significant publisher
bias (Friedman chi-squared = 42.73, p < 0.001).

**What I'd bring as a Research Scientist at Patronus.**

- **Redteaming research chops.** Adversarial taxonomies, multi-turn
  escalation analysis, defense-effectiveness ablations (dual-filter
  cuts harm rate 21.8% -> 4.8%, a 78% reduction).
- **Automated-eval instincts.** LLM-as-judge, meta-classifier stacking,
  eval reliability measurement (alpha, pairwise correlation, HDI on
  judge disagreement).
- **Statistical spine.** MS in Applied Statistics at RIT (2026);
  Bayesian hierarchical modeling, hypothesis testing with FDR/Bonferroni
  correction, calibration diagnostics.
- **Reproducibility habit.** MLflow lineage, model cards, IEEE
  2830-2025 audit trails - so results survive customer scrutiny, not
  just internal review.

I'm available for US remote and targeting a 2026 start once I finish
my MS. Portfolio, code, and three technical reports are on GitHub at
dl1413. The Red-Team framework is probably the fastest way to see
how I think about the eval-and-alignment problem Patronus is solving.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
