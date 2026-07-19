# Job Applications — 2026-07-19

**Target:** 5 roles per day, NYC or remote, anchored on the 3 primary projects:
- **AI Safety Red-Team Evaluation** (dual-stage LLM ensemble, 96.8% acc, α = 0.81, 340× cost reduction)
- **LLM Ensemble Bias Detection** (Bayesian hierarchical, α = 0.84, R-hat < 1.01, χ² = 42.73, p < 0.001)
- **Breast Cancer ML Classification** (99.12% acc, 100% precision, ECE 0.0089, SHAP + FastAPI)

All 5 postings below are live on Greenhouse as of today. Apply via the linked application form. Cover letters are ready to paste; tailor the greeting/hiring-manager name if the posting names one.

---

## 1. EvolutionIQ — Senior Data Scientist, LLM Evaluation (Medhub)

- **Location:** New York, NY or Remote
- **Comp:** $200K–$240K base + bonus + RSUs
- **Apply:** https://job-boards.greenhouse.io/evolutioniq/jobs/5748219004
- **Why it fits (top 1):** The role literally is what the Red-Team and Bias Detection projects already do — designing LLM evaluation scorecards, quantifying inter-rater reliability (Cohen's / Fleiss' / Krippendorff's kappa), and giving Go/No-Go recommendations from statistical confidence intervals.

### Cover letter

> Dear EvolutionIQ Hiring Team,
>
> The Medhub role reads like the job description for two projects I've already shipped. In my AI Safety Red-Team Evaluation, I stood up a dual-stage LLM-ensemble framework (GPT-4o, Claude-3.5, Llama-3.2 → Stacking Classifier) that scored 12,500 response pairs across six harm categories at 96.8% accuracy while holding Krippendorff's α at 0.81 — the audit-grade reliability threshold. The pipeline replaced a $6.12/sample human annotation baseline with a $0.018/sample automated one (340× cost reduction) and produced explicit Go/No-Go decision boundaries via bootstrap CIs and calibrated probabilities. That is exactly the "AI equivalent of a Quality Engineer" pattern you describe.
>
> The companion project — LLM Ensemble Textbook Bias Detection — pushed the statistical side further: 67,500 ratings across 4,500 passages, PyMC hierarchical model with partial pooling and MCMC convergence (R-hat < 1.01), Friedman χ² = 42.73 (p < 0.001), and Spearman-based publisher correlation matrices to catch structural agreement patterns human reviewers miss. I lean heavily on Krippendorff's α, Cohen's κ, multiple-testing correction (Bonferroni / FDR), and 95% HDIs to defend a result — the "Data Scientist's Data Scientist" mindset the posting asks for.
>
> Applied Statistics MS from RIT (Bayesian methods, experimental design), Python (Pandas / statsmodels / PyMC / scikit-learn), SQL, MLflow, and three publication-grade technical reports with model cards and calibration plots. Portfolio: https://github.com/dl1413 · https://dl1413.github.io/LLM-Portfolio/.
>
> Would love to talk about how I'd design the Medhub scorecard architecture for extraction, summarization, and chat evaluation.
>
> Best,
> Derek Lankeaux

---

## 2. Anthropic — Data Scientist, Policy

- **Location:** San Francisco, CA or New York, NY
- **Apply:** https://job-boards.greenhouse.io/anthropic/jobs/5232055008
- **Why it fits:** IEEE 2830-2025, ISO/IEC 23894:2025, and EU AI Act alignment show up in every published report. Bayesian hierarchical modeling and inter-rater reliability translate directly to policy analysis of frontier-model impact.

### Cover letter

> Dear Anthropic Policy Team,
>
> Anthropic's policy work sits at the intersection of frontier-model measurement and the governance frameworks the rest of the world uses to reason about AI risk — and that intersection is where I've deliberately built my portfolio. Every one of my three published technical reports is aligned with IEEE 2830-2025 (Transparent ML), ISO/IEC 23894:2025 (AI Risk Management), and the EU AI Act, with the artifacts (model cards, SHAP explanations, calibration plots, audit trails) that regulators and internal reviewers actually need.
>
> The empirical substrate is real work, not slideware. My AI Safety Red-Team Evaluation used a GPT-4o / Claude-3.5 / Llama-3.2 ensemble to score 12,500 response pairs across a MITRE ATLAS-aligned taxonomy, quantifying that multi-turn escalation is the single highest-risk attack vector (31.8% of surviving harms) and that a dual-filter defense cuts harm rate from 21.8% to 4.8% (78% reduction) — the kind of quantified before/after policymakers need to weigh mitigations. My LLM Bias Detection project used a PyMC hierarchical model (R-hat < 1.01, 95% HDI, Friedman χ² = 42.73, p < 0.001) to detect credible bias in 3 of 5 publishers analyzed — with Bayesian uncertainty quantification so we don't overclaim.
>
> I bring an Applied Statistics MS (RIT, Bayesian methods, causal inference), fluency across the frontier-model APIs, and the ability to translate a statistical finding into a memo a non-technical policymaker can act on. Portfolio: https://github.com/dl1413 · https://dl1413.github.io/LLM-Portfolio/.
>
> Excited to help sharpen how Anthropic quantifies the claims it makes to the outside world.
>
> Best,
> Derek Lankeaux

---

## 3. Garner Health — Data Scientist II (Health Intelligence)

- **Location:** New York, NY (in-office Tue/Wed/Thu)
- **Comp:** $150K–$175K + equity
- **Apply:** https://job-boards.greenhouse.io/garnerhealth/jobs/5654858004
- **Why it fits:** Clinical predictive modeling with statistical rigor, SHAP for provider transparency, calibrated decision thresholds — the exact toolkit from the Breast Cancer project applied to provider performance.

### Cover letter

> Dear Garner Health Intelligence Team,
>
> Provider performance modeling is a decision-support problem: the model has to be accurate, calibrated, explainable, and defensible to clinicians who will push back on any rule they can't interpret. That's the loop I ran end-to-end on my Breast Cancer ML Classification project — an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) that landed at 99.12% accuracy, 100% precision, and ROC-AUC 0.9987, with Platt scaling driving ECE from 0.0312 to 0.0089 so the probabilities were clinically usable. I tuned context-adaptive thresholds (100% sensitivity at 0.31 for screening) and shipped SHAP explanations plus a FastAPI service under 100ms p95 — the same shape as a provider-facing data product.
>
> The statistical rigor Garner asks for is where I lean hardest. My Bayesian work — PyMC hierarchical modeling with partial pooling, MCMC convergence (R-hat < 1.01), 95% HDIs, and multiple-testing correction (Bonferroni / FDR) — is exactly the toolkit for defending "provider X performs better than provider Y" against senior stakeholders who need to know how sure we are. I've processed 80K+ API calls and 2.5M tokens in production pipelines with circuit breakers, exponential backoff, and MLflow tracking, so the engineering hygiene is there too.
>
> Applied Statistics MS (RIT, 2026), Python / SQL / Pandas / scikit-learn / PyMC, and three publication-grade reports written for both technical reviewers and non-technical partners. Comfortable with the Tue/Wed/Thu NYC cadence. Portfolio: https://github.com/dl1413 · https://dl1413.github.io/LLM-Portfolio/.
>
> Would love to talk about how the calibration and threshold-tuning patterns from the clinical project map onto provider performance rules.
>
> Best,
> Derek Lankeaux

---

## 4. Swayable — Research Data Scientist

- **Location:** Remote (US) / San Francisco
- **Apply:** https://job-boards.greenhouse.io/swayable/jobs/5056114007
- **Why it fits:** Causal inference on media content at scale — the Bias Detection project already does hierarchical Bayesian causal-style comparisons across publishers and topics, and the Red-Team work already prototypes methodology-to-product.

### Cover letter

> Dear Swayable Team,
>
> Measuring how media content actually shifts opinion is one of the few applied-statistics problems where hierarchical Bayesian modeling, careful experimental design, and stakeholder-friendly readouts all have to land at once — and it's the shape of problem I've been building for.
>
> My LLM Ensemble Bias Detection project is the closest structural match. I ran 67,500 ratings across 4,500 passages from five publishers, then fit a PyMC hierarchical model with partial pooling to get publisher-level posteriors with 95% HDIs (MCMC R-hat < 1.01, Friedman χ² = 42.73, p < 0.001, 3/5 publishers credibly biased). The Spearman-based inter-publisher correlation matrix surfaced editorial-relationship structure (ρ up to 0.74) that would have been invisible without the hierarchical view — the kind of "unexpected structure in messy human-labeled data" finding Swayable's platform lives on. Bootstrap-based passage-level uncertainty flagged the 12.3% high-uncertainty items for expert re-review, which is basically the workflow you'd want for high-stakes creative testing.
>
> On the engineering side: production-quality Python (Pandas / Polars / PyMC / statsmodels / scikit-learn), FastAPI + Docker services under 100ms p95, MLflow tracking, and comfort iterating in a startup rhythm. Applied Statistics MS from RIT (Bayesian methods, experimental design, causal inference). Three publication-grade technical reports. Portfolio: https://github.com/dl1413 · https://dl1413.github.io/LLM-Portfolio/.
>
> Excited by the possibility of pushing the causal-inference methodology forward on real consumer-insights data.
>
> Best,
> Derek Lankeaux

---

## 5. Garner Health — Data Scientist II, Product

- **Location:** New York, NY (in-office Tue/Wed/Thu)
- **Apply:** https://job-boards.greenhouse.io/garnerhealth/jobs/5806406004
- **Why it fits:** Product-side complement to #3 — the same clinical-decision-modeling toolkit, with more experimentation and stakeholder-communication surface area.

### Cover letter

> Dear Garner Health Product Team,
>
> The Data Scientist II, Product role is where the modeling toolkit I've been building meets the experimentation and stakeholder-communication surface I want to spend more of my time on.
>
> My Breast Cancer ML Classification project ran the full loop a product-facing DS role asks for: an 8-algorithm benchmark (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting) landing at 99.12% accuracy and ROC-AUC 0.9987, Optuna Bayesian hyperparameter search converging in 5× fewer trials than grid, Platt calibration bringing ECE from 0.0312 to 0.0089, and context-adaptive thresholds (100% sensitivity at 0.31 for screening) — every one of those steps was written up in a report a non-technical partner could act on. The FastAPI deployment (<100ms p95, MLflow registry) is the "get it in front of a real user quickly" step.
>
> On the experimentation side, my Bias Detection project used a hierarchical Bayesian design (partial pooling, R-hat < 1.01, 95% HDIs, Friedman χ² = 42.73, p < 0.001) with Bonferroni / FDR multiple-testing correction — the design pattern for defending A/B and multi-arm results against senior review. My AI Safety Red-Team work quantified a dual-filter defense at 78% harm reduction, framed as the concrete before/after a PM wants when deciding whether to ship.
>
> Applied Statistics MS (RIT, 2026), Python / SQL / Pandas / scikit-learn / PyMC / MLflow / FastAPI, three publication-grade reports, and comfort with the NYC Tue/Wed/Thu cadence. Portfolio: https://github.com/dl1413 · https://dl1413.github.io/LLM-Portfolio/.
>
> Excited to talk about how the calibration, threshold, and hierarchical-Bayesian patterns from the portfolio would land on the product side of Garner.
>
> Best,
> Derek Lankeaux

---

## Application checklist

- [ ] EvolutionIQ — Senior DS, LLM Evaluation (Medhub) — https://job-boards.greenhouse.io/evolutioniq/jobs/5748219004
- [ ] Anthropic — Data Scientist, Policy — https://job-boards.greenhouse.io/anthropic/jobs/5232055008
- [ ] Garner Health — Data Scientist II (Intelligence) — https://job-boards.greenhouse.io/garnerhealth/jobs/5654858004
- [ ] Swayable — Research Data Scientist — https://job-boards.greenhouse.io/swayable/jobs/5056114007
- [ ] Garner Health — Data Scientist II, Product — https://job-boards.greenhouse.io/garnerhealth/jobs/5806406004

## Notes for tomorrow

- Rotate away from Greenhouse-only. Add Lever and Ashby postings for source diversity.
- Anthropic has ~5 other DS roles (GTM, Supply, Developer Productivity, Platform Product, Lead Platform Product). Keep one Anthropic role per day at most.
- All 5 apps here need the same resume PDF — export `Resume_Derek_Lankeaux.md` to PDF once and reuse.
