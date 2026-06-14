# Daily Queue — 2026-06-14 (NYC + Remote, 5 apps)

**Owner:** Derek Lankeaux
**Source pipelines today:** Anthropic careers, Greenhouse/Lever boards, BuiltIn NYC, Two Sigma careers.
**Projects in rotation:**
1. AI Safety Red-Team Evaluation (lead for safety/eval/research)
2. LLM Ensemble Textbook Bias Detection (lead for Bayesian / T&S / DS)
3. Clinical-Grade Breast Cancer Classification (lead for healthcare / applied MLE)

All 5 packets below pull verbatim from `APPLICATION_SNIPPETS.md`. Action: paste
into the JD's resume builder / cover letter box, run the **Application Quality
Checklist** in `JOB_APPLICATIONS_PLAYBOOK.md` §5, submit, flip the tracker row
from `drafted` to `submitted`.

---

## 1. Anthropic — Research Engineer, Evals  (NYC / SF / Remote)

- **Role family:** AI Safety / LLM Evaluation
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (LLM Bias) for Bayesian rigor
- **JD link:** https://www.anthropic.com/jobs  (filter: "Evals" / "Research Engineer")
- **Resume bullets to lead:** §"Resume bullets — AI Safety / Red-Team / Alignment" (4 bullets)
- **Cover-letter opener (paste verbatim):**
  > Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that detects
  > harmful AI outputs at 96.8% accuracy and 340x lower cost than human
  > annotation, while preserving audit-grade reliability (Krippendorff's
  > alpha = 0.81).
- **Cover-letter body:** §"Cover letter paragraph — AI Safety lab (Anthropic / OpenAI / METR / Apollo)"
- **JD keywords to echo (3+):** red-team, eval harness, Constitutional AI, jailbreak, RLHF data quality
- **Notes:** Lead with the 340x cost-reduction hook; Anthropic prizes calibration / inter-rater work — front-load Krippendorff alpha = 0.81 and the PyMC 95% HDI line.

---

## 2. Prolific — AI Research Engineer  (Remote)

- **Role family:** LLM Evaluation / Eval Infra
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (LLM Bias) — same ensemble pattern at scale
- **JD link:** https://job-boards.greenhouse.io/prolific/jobs (search "AI Research Engineer")
- **Resume bullets to lead:** §"Resume bullets — LLM Evaluation / Eval Infra roles" (3 bullets)
- **Cover-letter opener:** same 340x hook as Anthropic
- **Cover-letter body:** §"Cover letter paragraph — Eval platform (Patronus / Galileo / Arize)"  (swap "Patronus" for "Prolific")
- **JD keywords to echo:** evaluation methodology, LLM-as-judge, agent eval, novel evaluation frameworks
- **Notes:** Prolific is human-data + eval; emphasize the *human-LLM cost ratio* (340x) and the 12,500-pair benchmark — that's the bridge from human evals to LLM evals their product solves.

---

## 3. Mistral AI — Research Engineer, Machine Learning  (Remote, EU/US ok)

- **Role family:** ML Research Engineer (eval-flavored)
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 3 (Breast Cancer) for production rigor
- **JD link:** https://jobs.lever.co/mistral/bada0014-0f32-4370-b55f-81c5595c7339
- **Resume bullets to lead:** §"Resume bullets — ML Research Engineer / Applied Research" (3 bullets)
- **Cover-letter opener:** 340x hook + a one-liner from Project 3 ("99.12% / ROC-AUC 0.9987 on a clinical-grade ensemble, deployed at <100ms p95")
- **Cover-letter body:** §"Cover letter paragraph — AI Safety lab" adapted: replace "AI Safety lab" framing with "research-to-production" framing (Mistral hires research engineers who streamline evaluation).
- **JD keywords to echo:** streamline evaluation, robust tooling, interface research with production, MLflow, throughput
- **Notes:** The JD explicitly mentions *streamlining evaluation* — echo that phrase verbatim alongside the **850 samples/hr** throughput number.

---

## 4. Flatiron Health — Machine Learning Engineer  (NYC, on-site/hybrid)

- **Role family:** Healthcare / Clinical ML
- **Lead project:** Project 3 — Clinical-Grade Breast Cancer Classification
- **Supporting:** Project 1 (rigor / audit trails)
- **JD link:** https://www.builtinnyc.com/job/machine-learning-engineer/2914468
- **Resume bullets to lead:** §"Resume bullets — Healthcare / Clinical ML" (3 bullets)
- **Cover-letter opener (paste verbatim):**
  > Trained an 8-algorithm benchmark ensemble that hits 99.12% accuracy,
  > 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987
  > on breast-cancer classification — above the 90–95% human-expert range
  > — and deployed it as a <100ms p95 FastAPI service.
- **Cover-letter body:** §"Cover letter paragraph — Healthcare ML role"
- **JD keywords to echo:** real-world evidence, oncology, clinical decision support, SHAP, model registry, EHR-adjacent
- **Notes:** Flatiron is oncology-EHR; the breast-cancer story is the cleanest fit in your portfolio. Mention IEEE 2830-2025 explicitly — it's a regulatory keyword they care about.

---

## 5. Two Sigma — Quantitative Researcher: Machine Learning  (NYC)

- **Role family:** Data Scientist (Bayesian / Inference) / Quant ML
- **Lead project:** Project 2 — LLM Ensemble Textbook Bias Detection (Bayesian rigor)
- **Supporting:** Project 3 (Breast Cancer) for ensemble / model selection chops
- **JD link:** https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Quantitative-Researcher-Machine-Learning/12634
- **Resume bullets to lead:** §"Resume bullets — Data Scientist (Bayesian / Causal)" (3 bullets)
- **Cover-letter opener (paste verbatim):**
  > Built a multi-LLM bias-rating pipeline that processed 67,500 ratings
  > over 4,500 passages (2.5M tokens) at Krippendorff's alpha = 0.84, and
  > found statistically significant publisher bias (Friedman chi-squared =
  > 42.73, p < 0.001) in 3 of 5 publishers.
- **Cover-letter body:** §"Cover letter paragraph — Data / Stats-heavy role"
- **JD keywords to echo:** noisy data, hierarchical modeling, signal extraction, statistical rigor, hypothesis testing
- **Notes:** Two Sigma JD emphasizes "deep learning techniques to many types of problems, particularly those with large amounts of noisy data" — echo "noisy data" verbatim; lead the *posterior inference / R-hat < 1.01* line because that demonstrates Bayesian fluency they screen for.

---

## Quality gate (run before each submit, per playbook §5)

- [ ] Lead project sits at top of Projects section in the resume
- [ ] Cover letter opens with a metric hook (340x / 99.12% / Krippendorff alpha)
- [ ] ≥3 JD phrases echoed verbatim
- [ ] LinkedIn / GitHub / portfolio URLs live
- [ ] Salary expectation: "open, targeting market for the role/location"
- [ ] Work authorization answered
- [ ] JD saved as PDF into this folder

## Logging

5 rows appended to `application_tracker.csv` with status `drafted`. Flip each
to `submitted` after hitting submit, and log a `next_action` (e.g. `wait 10d`).
