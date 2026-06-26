# Daily Application Batch — 2026-06-04 (Thu)

**Owner:** Derek Lankeaux
**Target:** 5 tailored applications, NYC or US-remote
**Project rotation today:** P1×3 (Safety/Eval), P2×1 (Bayesian DS), P3×1 (Clinical ML)

Pull from `APPLICATION_SNIPPETS.md` for the paragraphs referenced below. Before
submitting each, run the §5 quality checklist from `JOB_APPLICATIONS_PLAYBOOK.md`.

---

## 1. Anthropic — Research Engineer, Alignment Evaluations (Remote / NYC)

- **Tier / source:** A — `careers.anthropic.com` (filter: Research Engineer + Evaluations / Alignment)
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 (LLM Bias Detection), P3 (rigor / SHAP)
- **Resume order:** P1 → P2 → P3
- **Snippet to paste:** APPLICATION_SNIPPETS.md → "Cover letter paragraph — AI Safety lab (Anthropic / OpenAI / METR / Apollo)"
- **JD keywords to mirror (target ≥3 verbatim):** red-team, eval harness, Constitutional AI, Krippendorff's alpha, jailbreak, model behavior
- **Hook line:** *"3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) at 96.8% accuracy, 340x cost reduction, Krippendorff's alpha = 0.81."*
- **Open question to answer in app:** Why Anthropic specifically — cite a recent Anthropic Safety post or RSP update from the last 60 days.

---

## 2. Hugging Face — LLM Evaluation Engineer (Remote / NYC presence)

- **Tier / source:** B — `huggingface.co/jobs` (filter: Evaluation / Research Engineer)
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 (multi-LLM ensemble at scale)
- **Resume order:** P1 → P2 → P3
- **Snippet to paste:** APPLICATION_SNIPPETS.md → "Cover letter paragraph — Eval platform (Patronus / Galileo / Arize)" — swap company to Hugging Face, reference `lighteval` / `evaluate` library if JD mentions it
- **JD keywords to mirror:** eval harness, LLM-as-judge, open-source, leaderboard, reproducibility, prompt iteration
- **Hook line:** *"Stacking meta-classifier over GPT-4o / Claude-3.5 / Llama-3.2 judges at 850 samples/hr, $0.018/sample, MLflow lineage end-to-end."*
- **Portfolio note:** Link the published PDF (`AI_Safety_RedTeam_Evaluation_Publication.pdf`) — HF reviewers respect published artifacts.

---

## 3. Patronus AI — Evaluation Platform Engineer (Remote)

- **Tier / source:** B — `patronus.ai/careers`
- **Lead project:** P1 — AI Safety Red-Team Evaluation
- **Supporting:** P2 (production-scale eval throughput)
- **Resume order:** P1 → P2 → P3
- **Snippet to paste:** APPLICATION_SNIPPETS.md → "Cover letter paragraph — Eval platform" — swap company name; emphasize the meta-classifier idea (47 features over 3 judges) since that maps to their product surface
- **JD keywords to mirror:** evaluation, hallucination, safety guardrails, ensemble judges, customer-grade reliability, FastAPI
- **Hook line:** *"47-feature stacking meta-classifier reconciles disagreement between 3 frontier LLM judges with 95% HDI per-model blind-spot detection."*
- **Differentiator:** Lead with the Bayesian disagreement-modeling layer — that's the part most eval startups don't have.

---

## 4. Memorial Sloan Kettering — Computational Oncology / Clinical ML Engineer (NYC)

- **Tier / source:** D — `careers.mskcc.org` (filter: Computational Oncology / Machine Learning)
- **Lead project:** P3 — Clinical-Grade Breast Cancer ML Classification
- **Supporting:** P1 (rigor / auditability), P2 (statistical methodology)
- **Resume order:** P3 → P1 → P2
- **Snippet to paste:** APPLICATION_SNIPPETS.md → "Cover letter paragraph — Healthcare ML role" — fill in MSK's specific clinical-AI program (e.g., MSK Precision Pathology) in the closing sentence
- **JD keywords to mirror:** clinical decision support, SHAP, explainability, IEEE 2830-2025, calibration, multicollinearity (VIF), nested CV
- **Hook line:** *"99.12% accuracy, 100% precision (zero false positives), ROC-AUC 0.9987 — above the 90–95% human-expert range — deployed at <100ms p95."*
- **Compliance note:** Explicitly call out IEEE 2830-2025 + SHAP-per-prediction transparency; MSK reviewers screen for clinical-grade auditability.

---

## 5. Two Sigma — Data Scientist, Modeling (NYC)

- **Tier / source:** C — `twosigma.com/careers` (filter: Data Scientist / Modeling / Research)
- **Lead project:** P2 — LLM Ensemble Textbook Bias Detection
- **Supporting:** P1 (production eval infra), P3 (rigor)
- **Resume order:** P2 → P1 → P3
- **Snippet to paste:** APPLICATION_SNIPPETS.md → "Cover letter paragraph — Data / Stats-heavy role"
- **JD keywords to mirror:** Bayesian hierarchical, MCMC, R-hat, partial pooling, hypothesis testing, experimental design, PyMC
- **Hook line:** *"67,500 LLM ratings → PyMC partial-pooling hierarchy (R-hat < 1.01) → publisher bias p < 0.001 at Friedman chi-squared = 42.73."*
- **Quant tone:** Cover letter should read like a stats memo, not marketing copy — lead with the test statistic and convergence diagnostic, not the headline number.

---

## End-of-day log (fill at 5:00 PM)

- Apps submitted today: __ / 5
- Snags (JD pulled, recruiter screen booked, etc.): __
- Tomorrow's seed: rotate to 2× Tier C (finance) + 2× Tier E (NYC product) + 1× Tier A
