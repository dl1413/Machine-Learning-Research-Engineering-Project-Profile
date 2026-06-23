# Salesforce — Adversarial AI & Research Engineer

- **Location:** Remote (NY / WA / TX-Austin / CA-SF eligible)
- **Source:** careers.salesforce.com (jr332451)
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework
- **Supporting:** P2 (LLM Bias Detection), P3 (rigor)
- **Fit note:** JD lists 6+ yrs hands-on AI/ML security testing — this is a *stretch*.
  Apply, lead hard with red-team eval depth, and frame the MS + published independent
  research as substituting for tenure. Skip if a hard 6-yr filter blocks the form.
- **JD phrases to echo:** adversarial AI, LLM vulnerabilities, testing the security of
  AI/ML systems, adversarial prompts, model robustness, red team.

## Resume — Projects section order
1. AI Safety Red-Team Evaluation Framework (lead)
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer Classification

### Top resume bullets (paste at top of Projects)
- Engineered dual-stage LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Designed 47 linguistic/semantic/structural features capturing jailbreak, refusal-evasion, and policy-violation signals to probe **LLM vulnerabilities** and **model robustness**
- Cut human-eval cost 340x ($6.12 → $0.018/sample) while holding Krippendorff's alpha = 0.81 across raters
- Shipped IEEE 2830-2025-compliant audit pipeline with SHAP explainability and full provenance trails

### Skills to surface for this JD
LLM red-teaming, adversarial prompts, jailbreak detection, harm classification, LLM-as-judge, PyTorch, MLflow, FastAPI.

## Cover letter
> Dear Salesforce AI Security team,
>
> The work most relevant to your Adversarial AI & Research Engineer role is an
> independent AI Safety Red-Team Evaluation framework I built and published in
> January 2026. It ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges
> and trains a stacking classifier on 47 harm-signal features, reaching 96.8%
> accuracy and ROC-AUC 0.9923 against a 12,500-pair benchmark across 6 harm
> categories. The feature set exists to surface concrete **LLM vulnerabilities** —
> jailbreak, refusal-evasion, and policy-violation patterns — which is the core of
> **testing the security of AI/ML systems**.
>
> Just as relevant for adversarial work: the harness runs at 850 samples/hr for
> $0.018/sample (a 340x cost reduction vs human annotation) with circuit breakers
> and async batching, and I paired it with a PyMC Bayesian hierarchical model that
> produces 95% HDI risk intervals per judge — so I can quantify *how confident* an
> adversarial finding is, not just flag it. I shipped the whole thing under IEEE
> 2830-2025 audit-trail requirements.
>
> I'm finishing an MS in Applied Statistics at RIT (2026) and available for remote
> from the NY area. I know this role typically wants more tenure; I'd ask you to
> weigh the published red-team work as evidence of exactly the **adversarial AI**
> capability you're hiring for. Portfolio and the three technical reports are on my
> GitHub (dl1413). US work-authorized; salary open and targeting market for the role.
>
> Best, Derek Lankeaux · dlankeaux12@gmail.com · linkedin.com/in/derek-lankeaux
