# 04 — Scale AI (Scale Labs) · Research Scientist, Frontier Risk Evaluations

- **Date pulled:** 2026-06-15
- **Location:** San Francisco, CA / New York, NY (per Scale Labs board listing)
- **Source:** labs.scale.com/jobs
- **JD link:** https://labs.scale.com/jobs
- **Lead project:** Project 1 — AI Safety Red-Team Evaluation
- **Supporting:** Project 2 (multi-model reliability, hierarchical uncertainty)
- **Resume version to send:** `resume_v3_safety.pdf`
- **Cover letter:** Yes

## JD signals (from public Scale Labs descriptions)

> Frontier Risk Evaluations team: designs and runs evaluations measuring
> dangerous-capability uplift across frontier models. Related Scale Labs
> teams: Agent Robustness, AI Controls and Monitoring, Safety Post Training.

Keywords to echo: `frontier model`, `dangerous capability`, `red-team`,
`evaluation harness`, `harm taxonomy`, `LLM-as-judge`, `inter-rater
reliability`, `Constitutional AI`.

## Tailored cover-letter opener (paste-ready)

> Scale Labs' Frontier Risk Evaluations work maps directly onto what I've
> spent the last six months building. My AI Safety Red-Team Evaluation
> framework ensembles GPT-4o, Claude-3.5, and Llama-3.2 as red-team judges
> over a 12,500-pair benchmark across 6 harm categories — dangerous info,
> hate, deception, privacy, illegal activity, self-harm — and trains a
> stacking classifier on 47 engineered features (jailbreak signals,
> refusal-evasion, multi-turn escalation). The framework hits **96.8%
> accuracy, ROC-AUC 0.9923, and Krippendorff's α = 0.81** across raters,
> at **850 samples/hr and $0.018/sample (340× under human annotation)**.
> Multi-turn escalation came out as the highest-risk attack vector (31.8%
> success), which is exactly the dangerous-capability shape Scale's
> Frontier Risk team is calibrated to measure. I'd love to bring this to
> Scale Labs full-time.

## Resume bullets to surface

(from `APPLICATION_SNIPPETS.md` → Project 1 → Safety/Red-Team)

- Engineered dual-stage LLM ensemble auto-grading 12,500 response pairs across 6 harm categories at 96.8% accuracy, 97.2% precision, ROC-AUC 0.9923
- Cut human-eval cost 340× ($6.12 → $0.018/sample) while maintaining Krippendorff's α = 0.81 across GPT-4o, Claude-3.5, Llama-3.2 raters
- Designed 47 linguistic / semantic / structural features capturing jailbreak, refusal-evasion, and policy-violation signals
- Identified multi-turn escalation (31.8%) as highest-risk vector via MITRE ATLAS-aligned attack taxonomy

## Application checklist

- [x] Lead project surfaced first
- [x] 3+ JD phrases echoed (`frontier`, `red-team`, `evaluation`, `dangerous capability`)
- [x] Metric hook in opener (96.8%, 340×, α = 0.81)
- [x] MITRE ATLAS alignment called out (Scale-relevant)
- [x] Work-auth: US authorized
- [x] Salary expectation: open
