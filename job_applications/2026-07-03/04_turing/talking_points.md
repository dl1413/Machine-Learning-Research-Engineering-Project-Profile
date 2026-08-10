# Recruiter Call — Talking Points (Turing)

1. **Opening (30s):** "I have built the rubric-scored LLM-ensemble eval loop twice — once for textbook bias (67,500 ratings, α = 0.84) and once for AI safety (12,500 responses, α = 0.81). Both used Bayesian hierarchical modeling to route uncertainty back to human labelers. That is the Turing post-training loop, just in academic clothing."

2. **Post-training relevance:** "I have not personally run an RLHF training loop, but I have designed the eval and reward-signal side of it — which reward model do you trust, and how do you know its ranking is stable? That is what my Bias project was really about."

3. **Cost & scale:** "$0.018/sample and 850/hr with circuit breakers and exponential backoff. Turing sells hourly expert time — the ratio of expert hours to model-hours is a knob I have spent a lot of time turning."

4. **Where I would want to grow:** "The RLVR side — verifiable-reward loops in math and code — is where I have the least direct experience but the most curiosity. I would want a role where I can learn that from a team that has shipped it."

5. **Ask:** "Which client accounts have the deepest eval-methodology problems right now? I want to hear where the measurement is fuzzy, not where the training already works."
