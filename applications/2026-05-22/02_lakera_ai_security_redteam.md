# Lakera — AI Security Engineer, Red Team

- **Location:** United States, Remote
- **Source:** lakera.ai/careers · LinkedIn (4322959169)
- **Lead project:** P1 — AI Safety Red-Team Evaluation Framework
- **Supporting:** P2 (LLM Bias Detection)
- **Fit note:** Strong. Lakera is AI-security-native (prompt-injection / LLM firewall);
  the red-team eval harness is a direct signal.
- **JD phrases to echo:** AI security, red team, prompt injection, jailbreak, LLM
  security, adversarial prompts, model misalignment.

## Resume — Projects section order
1. AI Safety Red-Team Evaluation Framework (lead)
2. LLM Ensemble Textbook Bias Detection
3. Breast Cancer Classification

### Top resume bullets
- Built production eval harness processing 850 samples/hr with circuit breakers, exponential backoff, and MLflow run tracking to detect **jailbreak** and **prompt-injection**-style policy violations
- Stacked GPT-4o / Claude-3.5 / Llama-3.2 judges into an XGBoost meta-classifier reaching 96.8% agreement with gold human labels on 12,500 pairs
- Designed 47 harm-signal features across 6 categories; quantified per-model blind spots via Bayesian hierarchical modeling (95% HDI)
- Cut human-eval cost 340x while holding Krippendorff's alpha = 0.81

### Skills to surface
LLM security, red-teaming, prompt injection, jailbreak detection, adversarial prompts, LLM-as-judge, FastAPI, Docker.

## Cover letter
> Dear Lakera team,
>
> I recently shipped a 3-model LLM red-team eval harness — GPT-4o, Claude-3.5,
> Llama-3.2 — that auto-grades 12,500 response pairs at 96.8% accuracy and 850
> samples/hr, with circuit breakers, async batching, and MLflow tracking baked in.
> The 47-feature detection layer targets exactly the failure modes Lakera defends
> against: **jailbreak**, refusal-evasion, and policy-violation signals, the
> building blocks of **prompt injection** and **LLM security** testing.
>
> What I'd bring to your **red team**: a habit of making adversarial findings
> *defensible*. The harness reconciles disagreement across three judges with a
> stacking meta-classifier and surfaces per-model blind spots via a PyMC Bayesian
> hierarchical model (95% HDI), so a flagged vulnerability comes with a confidence
> interval, not just a label. Cost landed at $0.018/sample — a 340x reduction vs
> human review — which matters when you need to red-team at scale.
>
> I'm wrapping an MS in Applied Statistics at RIT (2026), US work-authorized, and
> available fully remote. Code and three technical reports live on my GitHub
> (dl1413); the red-team eval is the fastest way to see how I think about
> AI security. Salary open, targeting market for the role.
>
> Best, Derek Lankeaux · dlankeaux12@gmail.com · linkedin.com/in/derek-lankeaux
