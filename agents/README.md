# Research Radar Agent

**Research Radar** turns a recurring, everyday task—keeping up with relevant
machine-learning research—into a concise daily brief. It is deliberately
single-purpose: it finds recent primary sources, explains their relevance to a
data-science workflow, and proposes one bounded experiment. This keeps the
agent useful without giving it authority to publish, purchase, deploy, or make
high-stakes decisions.

The implementation uses the OpenAI Responses API because this is a small,
single-agent workflow. The hosted `web_search` tool supplies current public
sources; the model synthesizes them under the instruction contract in
[`research_radar.py`](./research_radar.py).

## Agent definition

| Component | Decision |
| --- | --- |
| **User task** | Produce a daily ML research brief for the user's active interests. |
| **Audience** | A data scientist researching LLM evaluation, RAG reliability, AI safety, Bayesian ML, and production ML. |
| **Model** | `gpt-5.4` (set in `MODEL`). |
| **State** | Stateless by design; save the Markdown output in the user's preferred notes system if historical tracking is needed. |
| **Success criteria** | A brief has the five required sections, 3–5 source-linked reading items when enough reliable material exists, explicit uncertainty, and a reproducible experiment. |

### Instructions

The complete system instructions are stored as the `INSTRUCTIONS` constant in
[`research_radar.py`](./research_radar.py). They require fresh web research,
primary-source preference, linked sources, a fixed Markdown structure, and a
clear distinction between evidence and interpretation. They also tell the
agent to report when the requested window does not contain enough credible
items instead of fabricating recommendations.

### Tools and guardrails

| Tool | Purpose | Constraint |
| --- | --- | --- |
| Hosted `web_search` | Find current papers, official reports, and documentation. | The agent must prioritize primary sources and list every URL it uses. |
| `OPENAI_API_KEY` | Authenticates the live Responses API request. | Read from the environment only; never add it to source control. |

The agent has **no write, shell, email, calendar, or purchasing tools**. That
least-privilege choice prevents a research assistant from taking unintended
actions. It does not replace peer review or domain-specific professional
advice.

## Run the complete workflow

1. Create and activate a virtual environment (recommended), then install the
   lightweight runtime dependency:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-agent.txt
   ```

2. Set an API key in your shell; do not place it in a committed `.env` file:

   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. Run a tailored, current brief:

   ```bash
   python agents/research_radar.py \
     --topics "RAG evaluation, uncertainty quantification, and LLM safety" \
     --days 7 \
     --reader "applied ML researcher"
   ```

4. Read the output before acting on it. Confirm that each reading item has a
   working primary-source link, that dates fit the requested window, and that
   the proposed experiment has both a baseline and metric.

## Test the workflow

The dry run is a complete offline smoke test: it validates inputs, renders the
same five-section output contract, and requires neither an API key nor network
access.

```bash
python agents/research_radar.py --dry-run --topics "RAG reliability" --days 7
python -m unittest discover -s tests -v
python scripts/validate_portfolio.py
```

For a live acceptance test, run the command in step 3 and check the review
criteria in step 4. The live response changes with the web, so it is assessed
by its source links and contract rather than exact text.
