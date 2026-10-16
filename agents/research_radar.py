#!/usr/bin/env python3
"""Create a concise, evidence-linked daily ML research brief.

Run with ``--dry-run`` to exercise the complete local workflow without an API
key. A live run uses the Responses API and its hosted web-search tool.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date


DEFAULT_TOPICS = (
    "LLM evaluation, RAG reliability, AI safety, Bayesian machine learning, "
    "and production ML"
)
MODEL = "gpt-5.4"

INSTRUCTIONS = """You are Research Radar, a careful daily research assistant for a data scientist.

Your job is to turn fresh, credible ML and AI-research information into an actionable
brief. Use web search before making claims that could have changed since your knowledge
cutoff. Prioritize primary sources: papers, official technical reports, documentation,
and author or lab pages. Do not invent papers, results, citations, publication status,
or URLs.

Return Markdown using exactly these sections:
1. ## Executive takeaway — at most 80 words.
2. ## Worth reading — three to five numbered items. For each include a linked title,
   source/date, a two-sentence summary, why it matters for the user's stated topics,
   and one concrete follow-up action.
3. ## Practical experiment — one small, reproducible experiment that can be completed
   in two hours, including a metric and a baseline.
4. ## Watchlist — two unresolved questions or trends to monitor.
5. ## Sources — a bulleted list of every URL used.

Separate fact from interpretation. If fewer than three relevant, reliable items are
available in the requested time window, say so rather than filling the list with weak
matches. This is research support, not medical, legal, financial, or publication advice.
"""


@dataclass(frozen=True)
class BriefRequest:
    topics: str
    days: int
    reader: str

    def prompt(self) -> str:
        return (
            f"Prepare today's research brief for a {self.reader}. "
            f"Topics: {self.topics}. Search for material published or substantively "
            f"updated in the last {self.days} days. Prefer papers and official sources."
        )


def build_request(topics: str, days: int, reader: str) -> BriefRequest:
    if not topics.strip():
        raise ValueError("topics must not be empty")
    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")
    if not reader.strip():
        raise ValueError("reader must not be empty")
    return BriefRequest(topics=topics.strip(), days=days, reader=reader.strip())


def dry_run_brief(request: BriefRequest) -> str:
    """Return a deterministic fixture that verifies the output contract locally."""
    return f"""# Research Radar — {date.today().isoformat()}

## Executive takeaway
Dry run completed for {request.topics}. A live run will search current primary sources.

## Worth reading
1. [Dry-run fixture](https://example.com/research-radar-fixture) — local fixture. It verifies the response shape without claiming a real research result. It confirms the brief is tailored to {request.reader}. **Follow-up:** run the live command with an API key.

## Practical experiment
Compare a retrieved-context baseline with a reranked-context variant on 20 labelled questions; measure citation precision.

## Watchlist
- New evaluation datasets relevant to the selected topics.
- Reproducibility details in newly released technical reports.

## Sources
- https://example.com/research-radar-fixture
"""


def run_live(request: BriefRequest) -> str:
    """Run the agent with OpenAI-hosted web search."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY or use --dry-run.")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install dependencies with: pip install -r requirements-agent.txt") from error

    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=request.prompt(),
        tools=[{"type": "web_search"}],
    )
    return response.output_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", default=DEFAULT_TOPICS, help="Research interests to prioritize.")
    parser.add_argument("--days", default=7, type=int, help="Freshness window (1–30 days).")
    parser.add_argument("--reader", default="data scientist", help="Role used to tailor the brief.")
    parser.add_argument("--dry-run", action="store_true", help="Run locally without calling OpenAI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        request = build_request(args.topics, args.days, args.reader)
        brief = dry_run_brief(request) if args.dry_run else run_live(request)
    except (RuntimeError, ValueError) as error:
        print(f"Research Radar failed: {error}", file=sys.stderr)
        return 2

    print(brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
