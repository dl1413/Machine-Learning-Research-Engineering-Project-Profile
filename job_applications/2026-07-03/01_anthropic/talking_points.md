# Recruiter Call — Talking Points (Anthropic)

1. **Opening (30s):** "I built the same eval loop Anthropic's Safety Evaluations team runs internally, at student scale — dual-stage LLM ensemble producing harm labels, feeding a classifier that hit 96.8% acc and α = 0.81 across three frontier models. I want to run that loop against Claude in production."

2. **Cost angle:** "$0.018 per labeled sample vs. $6.12 human. That ratio is the reason eval budgets can grow without human-annotation blowing up — and it's why I ended up caring about ensemble reliability first."

3. **Rigor angle:** "The number I am proudest of isn't 96.8% accuracy — it's Krippendorff's α = 0.81 on the raw ensemble labels before any classifier. That's what makes the pipeline auditable."

4. **RSP-adjacent framing:** "The multi-turn escalation finding (31.8% of confirmed harm) came out of Bayesian hierarchical modeling per model. That kind of per-model risk decomposition is what I'd want to bring to RSP threshold discussions."

5. **Ask:** "Which teams are actively staffing eval design vs. red-team execution vs. Applied AI? I have a clear preference for eval design but the red-team work matches too — I'd like to know where the biggest gap is."

## Anticipated tough questions

- **"You have no industry experience."** → Answer with cost, throughput, and reliability numbers (they're production-grade even if the context was academic).
- **"Why not a PhD?"** → Anthropic's own careers page states 50% of technical staff don't have PhDs. Frame the MS + three shipped projects as faster time-to-contribution.
- **"How would you evaluate Claude on X?"** → Have a 90-second answer for "design an eval for jailbreak robustness" ready — reuse Red-Team taxonomy.
