"""Offline checks for the cURL-only saved-agent session launcher."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "agents" / "new_agent_session.sh"


class NewAgentSessionTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_script_uses_saved_agent_and_session_overrides(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("agent_fdf14800ea9547f1bdc8e8a1ee3e257b35b6b092975a48a3a1", script)
        self.assertIn('model: "gpt-6-sol"', script)
        self.assertIn("Generate comprehensive documentation for a full-length heavy metal requiem LP", script)
        self.assertIn('reasoning: {effort: "medium", summary: "auto"}', script)
        self.assertIn('text: {format: {type: "text"}, verbosity: "medium"}', script)
        self.assertIn('environment: {type: "none"}', script)
        self.assertIn('OpenAI-Beta: agents=v1', script)

    def test_script_streams_and_returns_function_results(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--no-buffer", script)
        self.assertIn("events?stream=true", script)
        self.assertIn("agent.session.input.tool_result", script)
        self.assertIn("required_actions", script)


if __name__ == "__main__":
    unittest.main()
