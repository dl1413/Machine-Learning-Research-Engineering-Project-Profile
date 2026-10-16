"""Offline contract tests for the Research Radar workflow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))
import research_radar  # noqa: E402


class ResearchRadarTests(unittest.TestCase):
    def test_dry_run_has_all_required_sections(self) -> None:
        request = research_radar.build_request("RAG reliability", 7, "ML researcher")
        brief = research_radar.dry_run_brief(request)

        for heading in (
            "## Executive takeaway",
            "## Worth reading",
            "## Practical experiment",
            "## Watchlist",
            "## Sources",
        ):
            self.assertIn(heading, brief)
        self.assertIn("https://example.com/research-radar-fixture", brief)

    def test_request_rejects_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "topics"):
            research_radar.build_request(" ", 7, "researcher")
        with self.assertRaisesRegex(ValueError, "days"):
            research_radar.build_request("AI safety", 0, "researcher")
        with self.assertRaisesRegex(ValueError, "reader"):
            research_radar.build_request("AI safety", 7, " ")

    def test_prompt_carries_user_preferences(self) -> None:
        request = research_radar.build_request("Bayesian ML", 14, "statistician")
        prompt = request.prompt()

        self.assertIn("Bayesian ML", prompt)
        self.assertIn("14 days", prompt)
        self.assertIn("statistician", prompt)


if __name__ == "__main__":
    unittest.main()
