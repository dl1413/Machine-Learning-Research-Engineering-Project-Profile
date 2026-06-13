# Daily Application Packet — 2026-06-13 (Sat)

**Owner:** Derek Lankeaux
**Target:** 5 tailored applications, all NYC on-site/hybrid or US-remote
**Routine:** scheduled run — packet is paste-ready for the user to submit Mon morning
**Project rotation today:** Project 1 ×2 (Safety/Eval), Project 2 ×1 (Bayesian/Stats), Project 3 ×2 (Healthcare/MLE)

> **3 primary projects** referenced throughout:
> - **P1** — AI Safety Red-Team Evaluation (96.8% acc · 340× cost reduction · α=0.81)
> - **P2** — LLM Ensemble Bias Detection (67.5K ratings · α=0.84 · Friedman χ²=42.73, p<0.001)
> - **P3** — Breast Cancer Classification (99.12% acc · 100% precision · ROC-AUC 0.9987)

---

## Application 1 — Anthropic · Research Engineer, Model Evaluations

- **Location:** New York, NY / San Francisco / Remote
- **Comp:** $320k–$485k base
- **Source:** https://job-boards.greenhouse.io/anthropic/jobs/5198255008
- **Lead project:** **P1** (AI Safety Red-Team Evaluation)
- **Supporting:** P2 (Bayesian rigor), P3 (production engineering)
- **JD keywords to mirror:** evaluation infrastructure · capabilities measurement · live training checkpoints · defensible metrics · scale · model behavior

### Opener / one-line hook
> Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that auto-grades harmful AI outputs at **96.8% accuracy and 340× lower cost** than human annotation, holding Krippendorff's α = 0.81 across 12,500 response pairs.

### Resume bullets (use top 3)
- Engineered dual-stage LLM ensemble evaluating 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into a meta-classifier on 47 features; surfaced per-model blind spots via PyMC Bayesian hierarchical model (95% HDI, R-hat < 1.01)
- Cut eval cost 340× ($6.12 → $0.018/sample) at 850 samples/hr with circuit breakers, async batching, and MLflow lineage
- Shipped under IEEE 2830-2025 audit-trail requirements with SHAP explainability and full provenance

### Cover letter body
> My most relevant work for Anthropic's Model Evaluations team is an independent AI Safety Red-Team Evaluation framework I shipped earlier this year. It ensembles GPT-4o, Claude-3.5-Sonnet, and Llama-3.2 as red-team judges and trains a stacking meta-classifier on 47 harm-signal features, reaching 96.8% accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark across six harm categories. The eval harness runs at 850 samples/hr for $0.018/sample — a 340× cost reduction versus human annotation — while holding inter-rater reliability at Krippendorff's α = 0.81. I paired it with a PyMC Bayesian hierarchical model that produces 95% HDI risk intervals per judge to surface systematic blind spots, and instrumented the whole thing under IEEE 2830-2025 audit requirements. The combination — defensible metrics, production throughput, and uncertainty quantification — is exactly the shape of work your posting describes, and I'd love to bring it to Claude's eval suite.

---

## Application 2 — Scale AI · Model Evaluations (NYC / Hybrid)

- **Location:** New York, NY (hybrid)
- **Source:** https://scale.com/careers/4631848005
- **Lead project:** **P1** (AI Safety Red-Team Evaluation)
- **Supporting:** P2 (multi-LLM eval at scale)
- **JD keywords to mirror:** model evaluations · benchmarking · red-team · adversarial · throughput · evaluation pipelines

### Opener / one-line hook
> Shipped a production eval harness processing **850 samples/hr at $0.018/sample** — a 340× cost reduction vs. human annotation — that holds Krippendorff's α = 0.81 across a 3-model LLM ensemble on 12,500 red-team pairs.

### Resume bullets (use top 3)
- Built production eval harness: 850 samples/hr, circuit breakers, exponential backoff, MLflow run tracking — 80K+ API calls / 2.5M tokens without manual intervention
- Designed adversarial attack taxonomy (8 categories, MITRE ATLAS-aligned); multi-turn escalation identified as highest-risk vector (31.8%)
- Measured defense effectiveness: dual-filter cuts harm rate 21.8% → 4.8% (78% reduction) on held-out red-team set
- Operated GPT-4o + Claude-3.5 + Llama-3.2 ensemble at 67.5K ratings on a parallel bias-detection project, α = 0.84, p < 0.001

### Cover letter body
> Scale's red-team and eval work is exactly the surface area I've been building toward. I recently shipped an LLM-ensemble eval framework that auto-grades 12,500 response pairs across six harm categories — dangerous info, hate, deception, privacy, illegal activity, self-harm — at 96.8% accuracy and 850 samples/hr, for $0.018/sample (340× cheaper than the human-annotation baseline). The taxonomy is MITRE ATLAS-aligned across eight adversarial categories; multi-turn escalation surfaced as the dominant risk vector at 31.8%, and the dual-filter defense I tested drops harm rate from 21.8% to 4.8%. On a sibling project I scaled the same multi-LLM pattern to 67,500 ratings with Krippendorff's α = 0.84 and statistically significant findings (Friedman χ² = 42.73, p < 0.001). The throughput, the adversarial framing, and the audit hygiene all map directly to your eval mission.

---

## Application 3 — Hugging Face · Research Engineer (Remote)

- **Location:** Remote (NYC office available)
- **Source:** https://huggingface.co/jobs
- **Lead project:** **P2** (LLM Ensemble Bias Detection — Bayesian/Stats lead)
- **Supporting:** P1 (LLM eval), P3 (rigor)
- **JD keywords to mirror:** open-source · evaluation · benchmark · reproducible · multi-model · fairness

### Opener / one-line hook
> Ran 67,500 LLM ratings across 4,500 textbook passages through GPT-4o / Claude-3.5 / Llama-3.2 with Krippendorff's α = 0.84, then defended the findings with a PyMC hierarchical model (R-hat < 1.01) producing 95% HDI publisher-level credible intervals.

### Resume bullets (use top 3)
- Operated 3-LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) at production scale: 67,500 ratings, 2.5M tokens, full MLflow lineage, 92% pairwise correlation
- Fit PyMC Bayesian hierarchical model with partial pooling across publishers; MCMC R-hat < 1.01, ESS > 1000, 95% HDI credible intervals per publisher × topic
- Ran Friedman omnibus (χ² = 42.73, p < 0.001) + post-hoc Nemenyi pairwise; flagged 12.3% high-uncertainty passages for expert review via bootstrap CIs
- Published reproducible technical report with priors, sensitivity analysis, and full posterior visualizations

### Cover letter body
> Hugging Face's role as the open-source reference point for model evaluation is what makes this team interesting to me. The project I'd want to lead with is an LLM-ensemble bias-detection study: 4,500 textbook passages, 2.5M tokens, 67,500 ratings from GPT-4o / Claude-3.5 / Llama-3.2. The headline result — 3 of 5 publishers showing statistically significant directional bias (Friedman χ² = 42.73, p < 0.001) — only holds because the pipeline was built to defend it: Krippendorff's α = 0.84 across raters, 92% pairwise correlation, and a PyMC partial-pooling hierarchical model with R-hat < 1.01 producing 95% HDI credible intervals per publisher and per topic. I published the full methodology, priors, sensitivity analysis, and posteriors as a reproducible report, which is the part I'd want to bring to your evaluation work — eval results are only as useful as the rigor backing them.

---

## Application 4 — Memorial Sloan Kettering · Machine Learning Scientist / Engineer

- **Location:** New York, NY (hybrid)
- **Source:** https://careers.mskcc.org/career-areas/digital-informatics-technology-solutions/
- **Lead project:** **P3** (Breast Cancer Classification)
- **Supporting:** P1 (rigor + audit), P2 (Bayesian uncertainty)
- **JD keywords to mirror:** clinical decision support · model calibration · explainability · oncology · tabular ML · cross-validation

### Opener / one-line hook
> Trained an 8-algorithm ensemble for breast-cancer classification hitting **99.12% accuracy, 100% precision (zero false positives), 98.59% recall, ROC-AUC 0.9987** — above the 90–95% human-expert range — with SHAP per-prediction explanations and a <100ms p95 FastAPI service.

### Resume bullets (use top 3)
- Built clinical-grade classifier exceeding the 90–95% human-expert range: 99.12% accuracy, 100% precision, 98.59% recall, ROC-AUC 0.9987
- Calibrated probabilities with Platt scaling (ECE 0.0312 → 0.0089, a 71.5% reduction) and tuned context-adaptive thresholds (100% sensitivity at 0.31 for mass-screening policy)
- Benchmarked 8 algorithms (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting +2) under stratified CV; Optuna TPE converged in 45 trials vs. 240 for grid search
- Productionized winner behind FastAPI + MLflow registry at <100ms p95; SHAP attributions per prediction for IEEE 2830-2025 transparency

### Cover letter body
> The MSK work that lines up with what I've been building is a clinical-grade breast-cancer classifier I shipped this spring. I benchmarked eight algorithms end-to-end — Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting, plus two baselines — under stratified cross-validation, and the winning ensemble lands at 99.12% accuracy, 100% precision (zero false positives across the held-out set), 98.59% recall, and ROC-AUC 0.9987 — comfortably above the 90–95% range typically cited for human-expert reads. As important for clinical deployment: probabilities are Platt-calibrated (ECE reduced 71.5%, from 0.0312 to 0.0089), thresholds are tuned per decision context (100% sensitivity at 0.31 for mass screening), VIF prunes multicollinear features, and the FastAPI service serves under 100ms p95 with SHAP attributions per prediction. The whole pipeline is documented to IEEE 2830-2025 transparency standards. I'd love to bring the same calibration-and-explainability discipline to MSK's clinical decision-support work.

---

## Application 5 — Flatiron Health · Machine Learning / NLP Engineer

- **Location:** New York, NY (hybrid)
- **Source:** https://flatiron.com/careers
- **Lead project:** **P3** (Breast Cancer Classification — oncology/EHR fit)
- **Supporting:** P1 (LLM ensemble for unstructured text), P2 (Bayesian uncertainty for clinical evidence)
- **JD keywords to mirror:** real-world evidence · oncology · EHR · NLP · production model · regulatory

### Opener / one-line hook
> Shipped a clinical-grade ensemble (99.12% acc, 100% precision, ROC-AUC 0.9987) for an oncology-adjacent classification task and complementary LLM-ensemble pipelines that process 2.5M tokens with audit-grade reliability — the exact pattern Flatiron's RWE work needs.

### Resume bullets (use top 3)
- Built oncology classifier at 99.12% accuracy / 100% precision / ROC-AUC 0.9987; deployed as FastAPI service at <100ms p95 with MLflow registry
- Operated multi-LLM ensembles processing 2.5M tokens / 80K+ API calls with circuit breakers, exponential backoff, and MLflow lineage — applicable to EHR free-text extraction
- Quantified uncertainty with PyMC Bayesian hierarchical models (R-hat < 1.01, 95% HDI) to support defensible real-world evidence claims
- Documented work to IEEE 2830-2025 and ISO/IEC 23894:2025 standards for regulated-domain audits

### Cover letter body
> Flatiron's mission — turning oncology EHR data into trustworthy real-world evidence — sits right at the intersection of the three projects I've shipped this year. The most directly relevant is a clinical-grade classifier for breast-cancer histopathology: 99.12% accuracy, 100% precision, ROC-AUC 0.9987, Platt-calibrated probabilities (ECE 0.0089), and a FastAPI service under 100ms p95 with SHAP explanations per prediction. The piece that translates most cleanly to your NLP-on-EHR pipelines is a sibling project where I ran an LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) across 2.5M tokens with circuit breakers, exponential backoff, MLflow lineage, and Krippendorff's α = 0.84 reliability — plus a PyMC hierarchical model producing 95% HDI credible intervals so any downstream claim can be defended. Everything ships to IEEE 2830-2025 and ISO/IEC 23894:2025 audit requirements, which is the standard regulated-domain RWE work demands.

---

## Universal closer (paste at end of every cover letter)

> I'm based for New York City and open to remote, targeting a 2026 start as I wrap my MS in Applied Statistics at RIT. Portfolio, code, and the three technical reports referenced above are on my GitHub (dl1413). Happy to walk through any of them on a call.
>
> Best,
> Derek Lankeaux

---

## Submission checklist (per application)

- [ ] Resume tailored — lead-project bullets pasted in top section
- [ ] Cover letter — opener hook + body + closer
- [ ] 3+ exact phrases from JD echoed in cover letter
- [ ] JD saved as PDF in this folder
- [ ] Row appended to `application_tracker.csv`
- [ ] LinkedIn referral check before submit (search 1st/2nd-degree at company)
