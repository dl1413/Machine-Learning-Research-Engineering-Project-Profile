# Cover Letter - Anthropic, Research Engineer, Model Evaluations

**Location:** New York, NY / Remote
**Lead project:** AI Safety Red-Team Evaluation Framework
**JD link:** https://job-boards.greenhouse.io/anthropic/jobs/5198255008

---

Dear Anthropic Model Evaluations team,

The reason I'm writing is that my most recent project - an independent AI
Safety Red-Team Evaluation framework I published in April 2026 - is exactly
the shape of work your Model Evaluations team does. It ensembles GPT-4o,
Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking
classifier on 47 harm-signal features, reaching 96.8% accuracy and
ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories
(dangerous info, hate, deception, privacy, illegal activity, self-harm).

The pipeline runs at 850 samples/hr for $0.018/sample - a 340x cost
reduction versus human annotation - while holding inter-rater reliability
at Krippendorff's alpha = 0.81. I paired it with a PyMC Bayesian
hierarchical model that produces 95% HDI risk intervals per judge, so we
can distinguish "the model is unsafe here" from "the judges disagree
here." The whole thing ships under IEEE 2830-2025 audit-trail
requirements, which matches how Anthropic already treats model cards and
system-card evidence.

What I'd bring to the Evaluations team:

- **Eval infrastructure that scales.** Circuit breakers, exponential
  backoff, async batching, and MLflow lineage across every run - the
  scaffolding needed to trust a benchmark that keeps growing.
- **A Bayesian view of judge disagreement.** Partial pooling + 95% HDIs
  turn a noisy 3-LLM ensemble into a defensible signal, and let us know
  when we need a human rater versus when the ensemble is already
  decisive.
- **Statistical rigor.** MS in Applied Statistics at RIT (2026), plus
  two more published projects using MCMC convergence diagnostics,
  Friedman / Nemenyi, calibration, and multiple-testing correction.
- **Comfort with Claude specifically.** I've been using Claude-3.5 as a
  primary judge across two projects; I know its refusal, self-consistency,
  and constitutional-AI behaviors from the eval side, not just from
  demos.

I'm based in / available for New York City and open to remote, targeting
a 2026 start once I wrap my MS. Portfolio, code, and three technical
reports (Red-Team, Bias Detection, Clinical ML) are on GitHub at
dl1413. Happy to walk through any of them - the Red-Team eval is
probably the fastest way to see how I think about your team's problem.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) | [GitHub](https://github.com/dl1413) | [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
