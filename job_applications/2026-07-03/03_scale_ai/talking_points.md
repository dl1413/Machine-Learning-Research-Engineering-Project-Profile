# Recruiter Call — Talking Points (Scale AI)

1. **Opening (30s):** "SEAL is why I put my Bias Detection project into the public portfolio — it's a working proof that a 3-model LLM ensemble can be calibrated against contested human ground truth at α = 0.84, with per-item uncertainty routed to human review. I want to bring that framework to your leaderboards."

2. **Rubric design:** "I built the 6-category harm taxonomy from scratch, mapped it onto MITRE ATLAS, and iterated the prompts until inter-rater α crossed 0.80. That's the exact loop I would want to run for a new SEAL benchmark."

3. **Ranking stability:** "Bootstrap CIs on passage-level ratings, plus PyMC hierarchical modeling for publisher rank order — I've thought about the 'is this ranking stable' problem in the specific form SEAL needs."

4. **Cost angle (SEAL-flavored):** "340× cost reduction on the Red-Team pipeline. Scale runs at scale where that ratio compounds — I'd want to look at where the current SEAL eval loop still has human-in-the-loop hot spots."

5. **Ask:** "Which benchmark family has the biggest gap right now — capability, safety, agentic, or multimodal? I have preferences, but the interesting question is where the measurement is weakest."
