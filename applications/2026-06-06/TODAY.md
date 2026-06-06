# Today's 5 — 2026-06-06 (queue for Mon 2026-06-08)

NYC + remote only. Each target maps to one of the 3 primary projects (lead) plus the other two as supporting evidence. All snippets are paste-ready; only swap in 2–3 JD phrases verbatim before submitting.

---

## 1. Scale AI — ML Research Scientist / Research Engineer, LLM Evaluation
- **Location:** SF / NYC / Seattle / remote-friendly. Posted base $179.4K–$224.3K.
- **JD:** https://scale.com/careers/4528010005
- **Role family:** LLM Evaluation / Eval Infra
- **Lead project:** AI Safety Red-Team Evaluation (Project 1)
- **Supporting:** LLM Bias Detection (Project 2)
- **Why it fits:** They build LLM-as-judge / autorater frameworks; your 3-model ensemble + 47-feature stacking meta-classifier is exactly the shape of that work.

### Cover-letter opener (paste, then swap in JD phrasing)
> I recently shipped a 3-model LLM eval harness — GPT-4o, Claude-3.5, Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and 850 samples/hr, with circuit breakers, async batching, and MLflow tracking baked in. Cost per sample landed at $0.018, a 340× reduction versus human review. The interesting part for Scale is the stacking layer: a 47-feature meta-classifier that reconciles disagreement between the three judges and surfaces per-model blind spots via Bayesian hierarchical modeling. That maps directly to the LLM-as-judge / autorater work your Evaluation team is scaling.

### Resume bullets (top of Projects)
- Built production LLM eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow tracking on 12,500 response pairs
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier (XGBoost) reaching 96.8% agreement with gold human labels; Krippendorff's α = 0.81
- Quantified judge disagreement with PyMC Bayesian hierarchical model (95% HDI), surfacing systematic blind spots per model family

### JD phrases to echo
- "LLM-as-a-Judge" / "autorater"
- "evaluation methodology, metrics, and benchmarks"
- "critique, grade, and explain agent outputs"

---

## 2. OpenAI — Applied / Forward-Deployed Engineering (LLM)
- **Location:** NYC (3-day hybrid)
- **JD:** https://openai.com/careers/search/ (filter NYC; pick newest "Applied" or "Forward Deployed" with LLM eval scope)
- **Role family:** LLM Evaluation / Applied
- **Lead project:** AI Safety Red-Team (Project 1)
- **Supporting:** LLM Bias Detection (Project 2)
- **Why it fits:** Forward-deployed roles want someone who can stand up an eval pipeline against a customer's models in a week — that's exactly the 3-model harness you've already published.

### Cover-letter opener
> I built and published an independent AI-safety eval framework that ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges and trains a stacking classifier on 47 harm-signal features, reaching 96.8% accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm categories. The pipeline runs at 850 samples/hr for $0.018/sample — a 340× cost reduction versus human annotation — while holding inter-rater reliability at Krippendorff's α = 0.81. For an applied / forward-deployed role at OpenAI, I'd bring the same instinct: stand up the eval scaffolding fast, prove it with metrics, and ship it under audit-grade reproducibility.

### Resume bullets
- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340× ($6.12 → $0.018/sample) while maintaining Krippendorff's α = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability, MLflow tracking, and full provenance trails

### JD phrases to echo
- "evaluation" / "evals"
- "GPT-4 / GPT-4o" (use the exact model variant the JD names)
- "production" / "deployment"

---

## 3. Flatiron Health — Senior ML Engineer, AI Incubator
- **Location:** NYC HQ (hybrid, 3 office days). Posted base $139.2K–$208.8K.
- **JD:** https://www.builtinnyc.com/job/senior-machine-learning-engineer-ai-incubator/3986687
- **Role family:** Healthcare / Clinical ML
- **Lead project:** Breast Cancer Classification (Project 3)
- **Supporting:** AI Safety Red-Team (Project 1) — for the GenAI angle the Incubator wants
- **Why it fits:** Oncology data, prototyping + scaling AI, hybrid NYC — your clinical-grade classifier + Project 1's LLM-evaluation scaffolding cover both halves of the Incubator's mandate.

### Cover-letter opener
> The work most relevant to Flatiron is a clinical-grade breast-cancer classifier I shipped this year: I benchmarked 8 algorithms end-to-end and landed at 99.12% accuracy with 100% precision (zero false positives), 98.59% recall, and ROC-AUC 0.9987, comfortably above the 90–95% range typically cited for human expert reads. The pipeline ships with SHAP per-prediction explanations, VIF-pruned features, SMOTE class balancing, and a FastAPI service under 100ms p95, all aligned with IEEE 2830-2025 transparency standards. On the GenAI-incubator side, I've also published a 3-LLM eval harness (GPT-4o / Claude-3.5 / Llama-3.2) that grades 12,500 outputs at 96.8% accuracy and 340× lower cost — the kind of evaluation rigor I'd want sitting under any clinical LLM prototype.

### Resume bullets
- Built clinical-grade classifier exceeding the 90–95% human-expert range: 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987
- Productionized winning model behind FastAPI with MLflow model registry and <100ms p95 latency; SHAP explanations per prediction
- Stood up parallel LLM evaluation harness (3-model ensemble) grading 12,500 outputs at 96.8% accuracy / $0.018 per sample — applicable to clinical LLM prototypes

### JD phrases to echo
- "oncology" / "clinical research"
- "prototype" / "incubator" / "scale"
- "production" / "deployment"

---

## 4. Patronus AI — Research Scientist (Remote)
- **Location:** Remote
- **JD:** https://www.glassdoor.com/job-listing/research-scientist-patronus-ai-inc-JV_KO0,18_KE19,34.htm?jl=1009651542176 (cross-check on patronus.ai/careers)
- **Role family:** LLM Eval Platform
- **Lead project:** AI Safety Red-Team (Project 1)
- **Supporting:** LLM Bias Detection (Project 2)
- **Why it fits:** Patronus builds adversarial test cases and LLM benchmarking as a product — your 8-category MITRE-ATLAS-aligned attack taxonomy and stacking judge are directly aligned.

### Cover-letter opener
> I'm interested in Patronus because I've spent the last few months building exactly the kind of multi-LLM evaluation infrastructure your product abstracts. My AI Safety Red-Team framework ensembles GPT-4o, Claude-3.5, and Llama-3.2 as judges over 12,500 response pairs across 6 harm categories, then trains a 47-feature stacking classifier reaching 96.8% accuracy and ROC-AUC 0.9923. The MITRE-ATLAS-aligned attack taxonomy — multi-turn escalation surfaced as the highest-risk vector (31.8%) — is the kind of adversarial test-case generation Patronus customers need on tap, not as a one-off project. I'd love to bring that taxonomy and the Bayesian-hierarchical-judge-disagreement work into a product team.

### Resume bullets
- Built adversarial AI-safety eval pipeline: 8-category MITRE-ATLAS-aligned attack taxonomy across 12,500 pairs; identified multi-turn escalation as highest-risk vector (31.8%)
- 3-LLM judge ensemble (GPT-4o / Claude-3.5 / Llama-3.2) → 47-feature stacking meta-classifier; 96.8% accuracy, ROC-AUC 0.9923, Krippendorff's α = 0.81
- Modeled judge disagreement with PyMC Bayesian hierarchy producing 95% HDI per-model risk intervals

### JD phrases to echo
- "adversarial test cases" / "red team"
- "LLM benchmarking" / "evaluation"
- "robustness"

---

## 5. Hugging Face — ML Engineer Internship, LLM Evaluation (US Remote)
- **Location:** US Remote
- **JD:** https://apply.workable.com/huggingface/ (filter "LLM Evaluation, US Remote")
- **Role family:** LLM Evaluation / Research Engineer
- **Lead project:** LLM Bias Detection (Project 2)
- **Supporting:** AI Safety Red-Team (Project 1)
- **⚠️ Flag:** Posted as an internship. Apply only if you're open to internship-level comp / scope, or DM the team to ask whether a contractor or new-grad full-time conversion exists. Otherwise, swap in a different HF posting from their workable page.

### Cover-letter opener
> The Open LLM Leaderboard problem — producing fair, reproducible eval results that the community will actually trust — is exactly the shape of work I've been doing. My LLM-ensemble bias study ran 67,500 ratings through GPT-4o / Claude-3.5 / Llama-3.2 over 4,500 textbook passages (2.5M tokens) at Krippendorff's α = 0.84 and 92% pairwise correlation, surfacing statistically significant publisher bias (Friedman χ² = 42.73, p < 0.001) in 3 of 5 publishers. It only counts as a finding because the eval scaffolding — Bayesian hierarchical model with R-hat < 1.01, MLflow lineage, circuit-breakered async API layer — was built to defend it. I'd love to bring that reproducibility-first habit to the Leaderboard.

### Resume bullets
- Operated 3-LLM ensemble (GPT-4o / Claude-3.5 / Llama-3.2) at production scale: 67,500 ratings, 4,500 passages, 2.5M tokens, full MLflow lineage
- Held inter-rater reliability at Krippendorff's α = 0.84 with 92% pairwise correlation; surfaced publisher-level bias at p < 0.001 (Friedman χ² = 42.73)
- Fit PyMC Bayesian hierarchical model with partial pooling; MCMC R-hat < 1.01, 95% HDI per publisher and topic

### JD phrases to echo
- "Open LLM Leaderboard"
- "reproducible" / "trustworthy evaluation"
- "ecosystem" / "community"

---

## Backlog (flagged, do not apply without checking)

- **Anthropic — Red Team Engineer, Safeguards** ($300K–$320K). 🚫 Requires SF relocation; not NYC/remote. Move to Tier-A list only if you're open to relocating.
- **Memorial Sloan Kettering — Machine Learning Scientist** (NYC hybrid). 🚫 Job posting requires PhD + 3 years DL. Apply only if a recruiter referral can vouch for MS + portfolio equivalence; otherwise watch for MSK's Machine Learning *Engineer* (non-Scientist) reqs.

---

## Submission checklist (run for each of 1–5 before hitting submit)
- [ ] Resume bullets re-ordered so lead project sits at top of Projects section
- [ ] Cover letter opens with the metric hook above
- [ ] At least 3 JD phrases appear verbatim in resume + cover letter
- [ ] LinkedIn, GitHub (dl1413), portfolio links present and live
- [ ] If JD lists PyTorch / JAX / Ray / vLLM, it appears in skills
- [ ] Salary expectation answered ("open, targeting market for the role/location")
- [ ] Work authorization answered
- [ ] JD saved as PDF in `applications/2026-06-06/`
- [ ] Row appended to `application_tracker.csv` flipped from `queued` → `submitted`
