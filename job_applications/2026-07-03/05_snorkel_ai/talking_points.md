# Recruiter Call — Talking Points (Snorkel AI)

1. **Opening (30s):** "My Bias Detection project is basically a Snorkel labeling-function stack, done by hand. Three LLM labelers, 92% pairwise correlation, PyMC as the generative aggregation model, α = 0.84 on the aggregated label. I want to bring that instinct onto your platform."

2. **Uncertainty routing:** "Bootstrap CIs at the item level flagged 12.3% of passages for human relabel. That is the exact 'flag the noisy sources' loop Snorkel already ships — I would want to make it smarter, not reinvent it."

3. **Two-stage LF pattern:** "The Red-Team project runs LLM-ensemble labeling into a Stacking Classifier that outperforms either stage alone (96.8% acc, α = 0.81 on raw labels). That's a working proof that LFs-plus-model beats LFs-alone at scale."

4. **Cost:** "340× cheaper than human annotation and it stays audit-grade. Enterprise labeling economics are the pitch — I have the receipts."

5. **Ask:** "Which client segments have the noisiest label sources right now — regulated industries, agentic-eval data, multilingual? Where the noise is worst is where the LF paradigm has the most room to prove itself."
