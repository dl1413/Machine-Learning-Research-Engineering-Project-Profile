#!/usr/bin/env python3
"""
Build three publication-formatted project documents under projects/:

  01_AI_Safety_RedTeam_Evaluation_{Report.md,Publication.pdf}
  02_LLM_Ensemble_Bias_Detection_{Report.md,Publication.pdf}
  03_Breast_Cancer_Classification_{Report.md,Publication.pdf}

Each source report's metadata header is normalized to one consistent
template, then rendered with the existing WeasyPrint publication pipeline
(generate_publication_pdfs.generate_pdf): journal title block,
abstract/keywords styling, stripped TOC/author sections, Dublin Core metadata.

Files follow the repo convention: `*_Report.md` for source, `*_Publication.pdf`
for the rendered publication.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Reuse the renderer from the existing publication pipeline.
from generate_publication_pdfs import generate_pdf

ROOT = Path(__file__).parent.resolve()
OUT  = ROOT / "projects"
OUT.mkdir(exist_ok=True)

# ─── Shared header template ──────────────────────────────────────────────────
# NOTE: the "AI Standards Compliance" key is the one build_title_block() reads,
# so the standards line renders in the publication title block.

PRO_HEADER = (
    "# {title}\n\n"
    "**Project:** {subtitle}  \n"
    "**Author:** Derek Lankeaux, MS Applied Statistics  \n"
    "**Role:** Data Scientist | Applied Statistician  \n"
    "**Institution:** Rochester Institute of Technology  \n"
    "**Date:** April 2026  \n"
    "**Version:** {version}  \n"
    "**AI Standards Compliance:** IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act (2025)\n\n"
    "> **Data Science Focus:** This report documents an end-to-end data "
    "science project — problem framing, statistical methodology, results with "
    "quantified uncertainty, and stakeholder-ready deliverables — relevant to "
    "2026 Data Scientist roles (experimentation, Bayesian inference, predictive "
    "modeling, and responsible-AI practice).\n\n"
    "---\n"
)


def strip_old_header(text: str) -> str:
    """Drop the original metadata header block up to (but not including) `## Abstract`."""
    m = re.search(r"^## Abstract\s*$", text, flags=re.MULTILINE)
    return text[m.start():] if m else text


def normalize(src: Path, title: str, subtitle: str, version: str) -> str:
    """Header-normalize a single report (body unchanged)."""
    body = strip_old_header(src.read_text(encoding="utf-8"))
    return PRO_HEADER.format(title=title, subtitle=subtitle, version=version) + body


# ─── Render to source MD + publication PDF ───────────────────────────────────

def render(out_stem: str, md_text: str) -> None:
    """Write the source markdown as `<stem>_Report.md` and render the
    publication-formatted PDF as `<stem>_Publication.pdf`, matching the
    repository's established source/publication file convention."""
    md_path  = OUT / f"{out_stem}_Report.md"
    pdf_path = OUT / f"{out_stem}_Publication.pdf"
    md_path.write_text(md_text, encoding="utf-8")
    generate_pdf(str(md_path), str(pdf_path))
    print(f"  Source MD  : {md_path.relative_to(ROOT)}")
    print(f"  Publication: {pdf_path.relative_to(ROOT)} ({pdf_path.stat().st_size // 1024} KB)")


# ─── Report manifest ─────────────────────────────────────────────────────────

REPORTS = [
    {
        "src": ROOT / "AI Safety Red-Team Evaluation_ Technical Analysis Report.md",
        "stem": "01_AI_Safety_RedTeam_Evaluation",
        "title": "AI Safety Red-Team Evaluation: Technical Analysis Report",
        "subtitle": ("Automated Harm Detection Using LLM-Ensemble Annotation and "
                     "Bayesian ML Classification"),
        "version": "2.0.0",
    },
    {
        "src": ROOT / "LLM_Ensemble_Bias_Detection_Report.md",
        "stem": "02_LLM_Ensemble_Bias_Detection",
        "title": "LLM-Ensemble Textbook Bias Detection: Technical Analysis Report",
        "subtitle": ("Detecting Publisher Bias Using an LLM Ensemble and Bayesian "
                     "Hierarchical Inference"),
        "version": "4.0.0",
    },
    {
        "src": ROOT / "Breast_Cancer_Classification_Report.md",
        "stem": "03_Breast_Cancer_Classification",
        "title": "Calibrated Binary Classification (WDBC): Technical Analysis Report",
        "subtitle": ("Ensemble Benchmarking, Probability Calibration, and "
                     "Decision-Policy Tuning on the Wisconsin Diagnostic Breast "
                     "Cancer Dataset"),
        "version": "4.0.0",
    },
]


def main() -> int:
    print("=" * 70)
    print("Building publication documents -> projects/")
    print("=" * 70)
    for r in REPORTS:
        if not r["src"].exists():
            print(f"  !! source missing: {r['src'].name}")
            return 1
        md = normalize(r["src"], r["title"], r["subtitle"], r["version"])
        render(r["stem"], md)
    print("=" * 70)
    print(f"Done. {len(REPORTS)} publication documents rendered.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
