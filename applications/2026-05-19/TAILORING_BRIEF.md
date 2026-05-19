# Tailoring Brief — 2026-05-19 (Tue) | 5 Apps, NYC + Remote

Each app leads with one of the three projects and lists the other two as
supporting evidence (per `JOB_APPLICATIONS_PLAYBOOK.md` §1). Before
submitting, paste the live JD URL into `application_tracker.csv` and confirm
3+ JD phrases appear verbatim in your resume + cover letter.

---

## 1. OpenAI — Member of Technical Staff, Evaluations (NYC)

- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (multi-LLM eval at scale), Project 3 (rigor / benchmarking)
- **Resume bullet set:** `APPLICATION_SNIPPETS.md` → Project 1 → "LLM Evaluation / Eval Infra"
- **Cover paragraph:** Project 1 → "AI Safety lab" variant; swap "[Company]" → "OpenAI", "[team]" → "Evaluations / Preparedness"
- **Hook:** *96.8% accuracy, ROC-AUC 0.9923, $0.018/sample (340x cost reduction), Krippendorff α = 0.81*
- **JD phrases to mirror (verify live):** "evaluations", "model behavior", "red-team", "LLM-as-judge", "harness"

## 2. Scale AI — Research Engineer, Red Team (Remote)

- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (bias / fairness signals), Project 3 (production deployment)
- **Resume bullet set:** Project 1 → "AI Safety / Red-Team / Alignment"
- **Cover paragraph:** Project 1 → "AI Safety lab" variant; emphasize the 47-feature meta-classifier and SHAP audit trail
- **Hook:** *3-model ensemble (GPT-4o / Claude-3.5 / Llama-3.2), 12,500 pairs, 6 harm categories, 850 samples/hr*
- **JD phrases to mirror:** "red-teaming", "jailbreak", "harm taxonomy", "human-AI agreement", "RLHF data quality"

## 3. Flatiron Health — Senior Machine Learning Engineer (NYC)

- **Lead project:** Project 3 — Clinical-Grade Breast Cancer Classification
- **Supporting:** Project 1 (audit-grade rigor), Project 2 (statistical methodology)
- **Resume bullet set:** Project 3 → "Healthcare / Clinical ML"
- **Cover paragraph:** Project 3 → "Healthcare ML role" variant; swap "[Company's specific clinical-AI problem]" → "Flatiron's oncology RWE pipelines"
- **Hook:** *99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987, <100ms p95 FastAPI*
- **JD phrases to mirror:** "clinical", "oncology", "tabular", "production model", "SHAP", "model registry"

## 4. Bloomberg AI — ML Research Engineer, NLP (NYC)

- **Lead project:** Project 2 — LLM Ensemble Bias Detection
- **Supporting:** Project 1 (eval infra), Project 3 (modeling rigor)
- **Resume bullet set:** Project 2 → "Data Scientist (Bayesian / Causal)" + "LLM Eval / Research Engineer"
- **Cover paragraph:** Project 2 → "Data / Stats-heavy role" variant; swap "[Company]" → "Bloomberg" and reference financial-text fairness as a natural extension
- **Hook:** *67,500 ratings over 4,500 passages (2.5M tokens), Krippendorff α = 0.84, Friedman χ² = 42.73, p < 0.001, R-hat < 1.01*
- **JD phrases to mirror:** "Bayesian", "hierarchical", "NLP", "inference", "experimental design", "rubric"

## 5. Patronus AI — LLM Evaluation Engineer (Remote)

- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (multi-LLM rating at production scale), Project 3 (latency / deployment)
- **Resume bullet set:** Project 1 → "LLM Evaluation / Eval Infra"
- **Cover paragraph:** Project 1 → "Eval platform (Patronus / Galileo / Arize)" variant — already keyed to this company
- **Hook:** *850 samples/hr eval harness, circuit breakers + exponential backoff, MLflow lineage, 47-feature stacking meta-classifier*
- **JD phrases to mirror:** "eval harness", "LLM-as-judge", "guardrails", "regression testing", "observability"

---

## Pre-submit Checklist (per app)

- [ ] Lead project sits at top of Resume → Projects section
- [ ] Cover letter opens with the metric hook above
- [ ] 3+ JD phrases appear verbatim across resume + cover letter
- [ ] Live JD URL pasted into `application_tracker.csv`
- [ ] JD saved as PDF in this folder (`applications/2026-05-19/<company>-JD.pdf`)
- [ ] Status flipped from `draft` → `submitted` after send

## End-of-Day

- Flip all 5 tracker rows to `submitted`
- Set `next_action` to `wait 10d` and a follow-up date
- Friday: roll up the week per Playbook §6 (response rate by role family)
