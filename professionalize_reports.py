#!/usr/bin/env python3
"""
Professionalize the three project reports into a single new `projects/` folder.

For each source report this script:
  1. Parses the metadata header (title + ``**Key:** value`` lines).
  2. Re-emits a consistent, professional title block (uniform field order,
     Data-Science role framing, EU AI Act in the compliance line, internal
     notebook ``Source:`` line dropped).
  3. Writes the cleaned Markdown to projects/<slug>.md.
  4. Renders a publication PDF to projects/<slug>.pdf using the existing
     WeasyPrint pipeline in generate_publication_pdfs.py.

Usage:
    python professionalize_reports.py
"""

from pathlib import Path
import re

# Reuse the battle-tested rendering pipeline rather than duplicating it.
from generate_publication_pdfs import md_to_publication_html, PUBLICATION_CSS
from weasyprint import HTML, CSS


# ─── Report manifest ────────────────────────────────────────────────────────
# (source markdown on the main branch) -> (clean slug, version)

REPORTS = [
    {
        "src": "AI Safety Red-Team Evaluation_ Technical Analysis Report.md",
        "slug": "01_AI_Safety_RedTeam_Evaluation",
        "version": "2.0.0",
    },
    {
        "src": "LLM_Ensemble_Bias_Detection_Report.md",
        "slug": "02_LLM_Ensemble_Bias_Detection",
        "version": "4.0.0",
    },
    {
        "src": "Breast_Cancer_Classification_Report.md",
        "slug": "03_Breast_Cancer_Classification",
        "version": "4.0.0",
    },
]

OUT_DIR = Path(__file__).parent / "projects"

# Uniform professional framing applied to every report.
ROLE = "Data Scientist | Applied Statistician"
AUTHOR = "Derek Lankeaux, MS Applied Statistics"
INSTITUTION = "Rochester Institute of Technology"
DATE = "April 2026"
COMPLIANCE = "IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act (2025)"
FOCUS = (
    "**Data Science Focus:** This report documents an end-to-end data science "
    "project — problem framing, statistical methodology, results with quantified "
    "uncertainty, and stakeholder-ready deliverables — relevant to 2026 Data "
    "Scientist roles (experimentation, Bayesian inference, predictive modeling, "
    "and responsible-AI practice)."
)


def split_header(md_text: str):
    """Return (title, fields_dict, body) by splitting on the first '---' rule."""
    lines = md_text.split("\n")
    title = ""
    fields = {}
    body_start = len(lines)

    for i, line in enumerate(lines):
        s = line.strip()
        if s == "---":
            body_start = i + 1
            break
        if s.startswith("# ") and not title:
            title = s[2:].strip()
            continue
        m = re.match(r"\*\*(.+?):\*\*\s*(.*)", s)
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()

    body = "\n".join(lines[body_start:]).lstrip("\n")
    return title, fields, body


def build_professional_header(title: str, fields: dict, version: str) -> str:
    """Emit a uniform metadata block (two trailing spaces => hard line breaks)."""
    project = fields.get("Project", "")
    lines = [f"# {title}", ""]
    if project:
        lines.append(f"**Project:** {project}  ")
    lines.append(f"**Author:** {AUTHOR}  ")
    lines.append(f"**Role:** {ROLE}  ")
    lines.append(f"**Institution:** {INSTITUTION}  ")
    lines.append(f"**Date:** {DATE}  ")
    lines.append(f"**Version:** {version}  ")
    lines.append(f"**Standards Compliance:** {COMPLIANCE}")
    lines += ["", f"> {FOCUS}", "", "---", ""]
    return "\n".join(lines)


def professionalize(md_text: str, version: str) -> str:
    title, fields, body = split_header(md_text)
    header = build_professional_header(title, fields, version)
    return header + body.rstrip() + "\n"


def main():
    base = Path(__file__).parent
    OUT_DIR.mkdir(exist_ok=True)
    css = CSS(string=PUBLICATION_CSS)

    print("=" * 70)
    print("Professionalizing project reports -> projects/")
    print("=" * 70)

    ok = 0
    for r in REPORTS:
        src = base / r["src"]
        if not src.exists():
            print(f"  SKIP (not found): {r['src']}")
            continue

        raw = src.read_text(encoding="utf-8")
        clean_md = professionalize(raw, r["version"])

        md_out = OUT_DIR / f"{r['slug']}.md"
        md_out.write_text(clean_md, encoding="utf-8")
        print(f"  MD : {md_out.relative_to(base)}")

        try:
            html = md_to_publication_html(clean_md)
            pdf_out = OUT_DIR / f"{r['slug']}.pdf"
            HTML(string=html).write_pdf(
                str(pdf_out),
                stylesheets=[css],
                pdf_version="1.7",
                optimize_images=True,
                presentational_hints=True,
            )
            size_kb = pdf_out.stat().st_size / 1024
            print(f"  PDF: {pdf_out.relative_to(base)} ({size_kb:.0f} KB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  PDF ERROR for {r['slug']}: {e}")

    print("=" * 70)
    print(f"Done. {ok}/{len(REPORTS)} reports rendered to PDF in projects/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
