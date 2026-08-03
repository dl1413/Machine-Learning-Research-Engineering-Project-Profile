#!/usr/bin/env python3
"""Validate the portfolio's durable project inventory and local links.

This intentionally uses only the Python standard library so it can run in CI
without installing the PDF or ML toolchain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]

PROJECTS = (
    (
        "AI Safety Red-Team Evaluation",
        "AI Safety Red-Team Evaluation_ Technical Analysis Report.md",
        "AI_Safety_RedTeam_Evaluation_Publication.pdf",
    ),
    (
        "Breast Cancer Classification",
        "Breast_Cancer_Classification_Report.md",
        "Breast_Cancer_Classification_Publication.pdf",
    ),
    (
        "LLM Ensemble Bias Detection",
        "LLM_Ensemble_Bias_Detection_Report.md",
        "LLM_Bias_Detection_Publication.pdf",
    ),
    (
        "RAG Production Pipeline",
        "RAG_Project_Report.md",
        "RAG_Project_Publication.pdf",
    ),
)

LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def validate_required_artifacts(errors: list[str]) -> None:
    for name, report, publication in PROJECTS:
        for relative in (report, publication):
            path = ROOT / relative
            if not path.is_file():
                fail(f"{name}: missing {relative}", errors)
            elif path.stat().st_size == 0:
                fail(f"{name}: empty {relative}", errors)

        publication_path = ROOT / publication
        if publication_path.is_file():
            with publication_path.open("rb") as handle:
                header = handle.read(4)
            if header != b"%PDF":
                fail(f"{name}: {publication} is not a PDF", errors)


def validate_readme_inventory(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, report, publication in PROJECTS:
        if name not in readme:
            fail(f"README.md: missing project heading/content for {name}", errors)
        for relative in (report, publication):
            encoded = relative.replace(" ", "%20")
            if relative not in readme and encoded not in readme:
                fail(f"README.md: missing link to {relative}", errors)


def validate_local_links(errors: list[str]) -> None:
    # These are the repository's human-facing indexes. Report prose contains
    # code and example URLs, so it is intentionally not treated as an index.
    indexes = [ROOT / "README.md", *sorted((ROOT / "project_packages").glob("*/README.md"))]
    for index in indexes:
        text = index.read_text(encoding="utf-8")
        for raw_target in LOCAL_LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            relative = unquote(parsed.path)
            if not relative:
                continue
            if relative.startswith("/"):
                fail(f"{index.relative_to(ROOT)}: local link must be relative {raw_target}", errors)
                continue
            resolved = (index.parent / relative).resolve()
            if not resolved.is_relative_to(ROOT):
                fail(f"{index.relative_to(ROOT)}: local link escapes repository root {raw_target}", errors)
                continue
            if not resolved.exists():
                fail(f"{index.relative_to(ROOT)}: broken local link {raw_target}", errors)


def main() -> int:
    errors: list[str] = []
    validate_required_artifacts(errors)
    validate_readme_inventory(errors)
    validate_local_links(errors)

    if errors:
        print("Portfolio validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Portfolio validation passed: {len(PROJECTS)} projects and all index links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
