# Job Applications Playbook — 5/Day, NYC + Remote

**Owner:** Derek Lankeaux
**Cadence:** 5 tailored applications per day (Mon–Fri = 25/week, ~100/month)
**Target geos:** New York City (on-site/hybrid) and US-remote

---

## 1. Role Targeting Matrix

Map each project to the role types where it's the strongest signal. In every
application, lead with the project that best matches the JD, then list the
other two as supporting evidence.

| Role Family | Lead Project | Supporting Projects | Example Titles |
|---|---|---|---|
| **AI Safety / Red-Team / Alignment** | AI Safety Red-Team Evaluation | LLM Bias Detection, Breast Cancer (rigor) | AI Safety Engineer, Red-Team Engineer, Alignment Researcher, Policy Research Engineer |
| **LLM Evaluation / Eval Engineering** | AI Safety Red-Team Evaluation | LLM Bias Detection | LLM Evaluation Engineer, Model Behavior Engineer, Eval Infra Engineer |
| **ML Research Engineer** | AI Safety Red-Team Evaluation | LLM Bias Detection, Breast Cancer | ML Research Engineer, Research Engineer, Applied Research |
| **Applied ML Engineer / MLE** | Breast Cancer Classification | AI Safety Red-Team, LLM Bias Detection | ML Engineer, Applied ML, MLE II |
| **Healthcare / Clinical ML** | Breast Cancer Classification | AI Safety Red-Team (rigor) | Clinical ML Engineer, Healthcare AI, Computational Pathology |
| **Data Scientist (Bayesian/Causal)** | LLM Bias Detection | Breast Cancer, AI Safety Red-Team | DS — Inference, Statistician, Causal DS |
| **Trust & Safety / Policy ML** | AI Safety Red-Team Evaluation | LLM Bias Detection | T&S Engineer, Integrity ML, Content Policy |

---

## 2. Target Companies (NYC + Remote-Friendly)

Maintain ~80 active leads at any time. Rotate through these lists daily so you
don't re-apply to the same place twice.

### Tier A — Frontier Labs (apply direct, 1/week each)
- Anthropic (remote / NYC presence) — *strongest fit; lead with Red-Team*
- OpenAI (NYC + remote) — *Red-Team or Evals*
- Google DeepMind (NYC) — *Research Engineer*
- Meta FAIR / GenAI (NYC) — *Research Engineer*
- xAI (remote) — *Eng / Eval*
- Cohere (NYC + remote) — *Applied Research*
- Mistral, Inflection, AI21, Adept, Reka

### Tier B — AI Safety / Eval-Native (apply direct)
- Scale AI, Surge AI, Invisible — *Eval Ops / Red-Team*
- METR, Apollo Research, Redwood Research, ARC Evals — *AI Safety Research*
- Hugging Face (NYC + remote) — *Eval / Research Engineer*
- LangChain, LlamaIndex — *Eval tooling*
- Patronus AI, Arize, Weights & Biases, Galileo — *LLM eval platforms*

### Tier C — NYC Finance ML
- Two Sigma, Citadel, Jane Street, Hudson River Trading, Jump Trading
- Bridgewater, Renaissance, D. E. Shaw, Point72 Cubist
- Goldman Sachs / JPM AI Research, Bloomberg AI

### Tier D — NYC Healthcare ML
- Memorial Sloan Kettering — Computational Oncology
- Flatiron Health, Tempus, Recursion (NYC ops), Insitro
- Mount Sinai Hasso Plattner Institute, NYU Langone AI
- Verily, Hinge Health, Cohere Health

### Tier E — NYC + Remote Product/Platform
- Spotify, Etsy, Squarespace, Datadog, MongoDB, Peloton (NYC HQ)
- Stripe, Notion, Vercel, Replit, GitLab (remote)
- Runway, Pika, ElevenLabs, Suno (creative AI, NYC)

### Job Boards (daily 30-min scan)
- LinkedIn Jobs — filter: NYC + Remote, "ML Research Engineer" OR "LLM" OR "AI Safety"
- aisafety.com/jobs, 80000hours.org/job-board
- Wellfound (ex-AngelList), BuiltIn NYC, YC Work at a Startup
- ai-jobs.net, mlh.io/jobs, cleared-jobs (if applicable)

---

## 3. The Daily 90-Minute Routine

| Block | Time | What |
|---|---|---|
| **Scan** | 0:00–0:20 | Pull 8–10 fresh JDs from boards. Skim, drop to 5 best fits. |
| **Tailor** | 0:20–1:15 | For each of 5 roles: identify lead project, pick snippets from `APPLICATION_SNIPPETS.md`, paste into resume bullets + cover letter, swap keywords from the JD. |
| **Submit** | 1:15–1:25 | Submit. Save JD as PDF in `applications/YYYY-MM-DD/`. |
| **Log** | 1:25–1:30 | Append row to `application_tracker.csv`. |

**Rule:** every application must (1) lead with the matching project, (2) echo 3+
exact phrases from the JD, (3) include a one-line metric hook in the cover letter.

---

## 4. Keyword Bank (paste into resume/cover letter as JD requires)

Pull from this bank to match JD vocabulary. Don't fabricate — every term below
is backed by your existing work.

**LLM / Eval:** LLM evaluation, red-teaming, jailbreak detection, harm
classification, Constitutional AI, inter-rater reliability, Krippendorff's alpha,
prompt engineering, LLM-as-judge, eval harness, RLHF data quality.

**ML Eng:** ensemble methods, stacking classifier, XGBoost, LightGBM, gradient
boosting, feature engineering, SMOTE, RFE, hyperparameter tuning, cross-validation,
production ML pipeline, MLOps, MLflow, model registry, drift monitoring.

**Statistical:** Bayesian hierarchical modeling, MCMC, R-hat convergence,
95% HDI, posterior inference, PyMC, partial pooling, Friedman test,
hypothesis testing, causal inference, experimental design.

**Infra:** FastAPI, Docker, Kubernetes, circuit breakers, exponential backoff,
rate limiting, async API integration, low-latency inference (<100ms p95),
850 samples/hr throughput.

**Compliance / Safety:** IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act,
SHAP explainability, fairness auditing, audit trails, model cards.

---

## 5. Application Quality Checklist

Before hitting submit, every app should pass:

- [ ] Resume bullet order rearranged so lead project sits at top of Projects section
- [ ] Cover letter opens with a metric hook (340x cost reduction / 99.12% accuracy / Krippendorff alpha = 0.84)
- [ ] At least 3 phrases from the JD appear verbatim in resume + cover letter
- [ ] LinkedIn URL, GitHub URL, Portfolio URL all live and current
- [ ] If JD lists a specific framework (PyTorch / JAX / Ray / vLLM), it appears in skills
- [ ] Salary expectation answered (default: "open, targeting market for the role/location")
- [ ] Work authorization answered

---

## 6. Weekly Review (Friday, 30 min)

- Count apps submitted (target: 25)
- Count callbacks / recruiter screens / rejections
- Compute response rate by **role family** — kill role families with <2% response rate after 50 apps
- Identify top 3 JD keywords from callback roles — add to next week's resume tweaks
- Refresh Tier A/B target list — remove filled roles, add new postings

---

## 7. Anti-Patterns to Avoid

- **Spray-and-pray with the same resume.** Every app must be tailored, even if
  only the top 3 bullets and cover letter intro change.
- **Applying to roles requiring >2 years more experience than you have.** Skip
  Staff/Principal unless JD explicitly invites MS-level applicants.
- **Ignoring the cover letter when "optional."** Optional cover letters are
  read first when present; submitting one is a 2x signal at frontier labs.
- **Applying without saving the JD.** JDs disappear; you'll need the text for
  interview prep.

See `APPLICATION_SNIPPETS.md` for ready-to-paste resume bullets and cover
letter blurbs keyed to each project.
