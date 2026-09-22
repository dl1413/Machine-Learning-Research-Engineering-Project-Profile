#!/usr/bin/env python3
"""Generate the one-page PDF résumé from its Markdown source."""

from pathlib import Path

import markdown
from weasyprint import CSS, HTML


ROOT = Path(__file__).parent
SOURCE = ROOT / "Resume_Derek_Lankeaux.md"
OUTPUT = ROOT / "Resume_Derek_Lankeaux.pdf"

RESUME_CSS = """
@page { size: letter; margin: 0.32in 0.45in; }
body {
    color: #171717;
    font-family: "Arial", "DejaVu Sans", sans-serif;
    font-size: 7.8pt;
    line-height: 1.18;
}
h1 {
    color: #0b3558;
    font-size: 18pt;
    letter-spacing: 0.2pt;
    line-height: 1;
    margin: 0 0 2pt;
}
h1 + p {
    color: #205b83;
    font-size: 9.4pt;
    font-weight: bold;
    margin: 0 0 2pt;
}
h1 + p + p {
    font-size: 7.7pt;
    margin: 0 0 7pt;
}
h2 {
    border-bottom: 0.8pt solid #205b83;
    color: #0b3558;
    font-size: 9.4pt;
    letter-spacing: 0.4pt;
    margin: 6pt 0 3pt;
    padding-bottom: 1.5pt;
    text-transform: uppercase;
}
h3 {
    color: #111;
    font-size: 8.4pt;
    margin: 4pt 0 1pt;
    page-break-after: avoid;
}
p { margin: 0 0 2pt; }
ul {
    margin: 1pt 0 2pt;
    padding-left: 12pt;
}
li {
    margin: 0 0 1pt;
    padding-left: 0;
}
strong { color: #111; }
a { color: #0b4f7d; text-decoration: none; }
table {
    border-collapse: collapse;
    font-size: 7.5pt;
    margin-top: 2pt;
    width: 100%;
}
th, td {
    border-bottom: 0.35pt solid #d5dde3;
    padding: 1.5pt 3pt;
    text-align: left;
}
th { color: #0b3558; }
"""


def build_html(source: str) -> str:
    """Convert the résumé Markdown into a compact, accessible HTML document."""
    body = markdown.markdown(source, extensions=["tables", "sane_lists"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Derek Lankeaux — Data Scientist Résumé</title>
  <meta name="author" content="Derek Lankeaux">
  <meta name="description" content="Data Scientist résumé tailored for healthcare AI and applied machine learning roles.">
</head>
<body>{body}</body>
</html>"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    HTML(string=build_html(source), base_url=str(ROOT)).write_pdf(
        OUTPUT, stylesheets=[CSS(string=RESUME_CSS)]
    )
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
