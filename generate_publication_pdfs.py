#!/usr/bin/env python3
"""
Generate publication-ready PDFs from project markdown reports.

Transforms markdown reports into academically formatted PDFs suitable for
publication, removing informal/resume sections and applying journal-style CSS.

Usage:
    python generate_publication_pdfs.py
"""

import re
import markdown
from weasyprint import HTML, CSS
from pathlib import Path


# ─── Publication CSS ────────────────────────────────────────────────────────

PUBLICATION_CSS = """
@page {
    size: letter;
    margin: 1in 0.85in 1in 0.85in;

    @bottom-center {
        content: counter(page);
        font-family: "Times New Roman", "DejaVu Serif", Georgia, serif;
        font-size: 10pt;
        color: #333;
    }
}

@page :first {
    @bottom-center { content: none; }
}

/* ── Base typography ── */
body {
    font-family: "Times New Roman", "DejaVu Serif", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    orphans: 3;
    widows: 3;
}

/* ── Title block ── */
.title-block {
    text-align: center;
    margin-bottom: 28pt;
    padding-bottom: 14pt;
    border-bottom: 0.75pt solid #333;
}

.title-block h1 {
    font-size: 17pt;
    font-weight: bold;
    margin: 0 0 10pt 0;
    line-height: 1.25;
    color: #000;
}

.title-block .author {
    font-size: 11.5pt;
    font-weight: bold;
    margin: 4pt 0 2pt 0;
}

.title-block .affiliation {
    font-size: 10pt;
    font-style: italic;
    color: #333;
    margin: 2pt 0;
}

.title-block .date {
    font-size: 10pt;
    color: #555;
    margin: 2pt 0;
}

.title-block .compliance {
    font-size: 8.5pt;
    color: #666;
    margin-top: 6pt;
}

/* ── Abstract ── */
.abstract {
    margin: 0 0.3in 18pt 0.3in;
    padding: 0;
}

.abstract h2 {
    font-size: 11pt;
    text-align: center;
    margin-bottom: 6pt;
}

.abstract p {
    font-size: 9.5pt;
    line-height: 1.35;
    text-indent: 0;
}

.keywords {
    font-size: 9pt;
    margin: 8pt 0.3in 18pt 0.3in;
    color: #333;
}

.keywords strong {
    color: #111;
}

/* ── Section headings ── */
h2 {
    font-size: 13pt;
    font-weight: bold;
    margin: 22pt 0 8pt 0;
    padding-bottom: 3pt;
    border-bottom: 0.5pt solid #999;
    color: #000;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: bold;
    margin: 16pt 0 6pt 0;
    color: #111;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 12pt 0 4pt 0;
    color: #222;
    page-break-after: avoid;
}

/* ── Paragraphs ── */
p {
    margin: 0 0 8pt 0;
    text-indent: 0;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

thead th {
    background-color: #f0f0f0;
    border-top: 1.5pt solid #333;
    border-bottom: 1pt solid #333;
    padding: 5pt 6pt;
    text-align: left;
    font-weight: bold;
    font-size: 8.5pt;
}

tbody td {
    border-bottom: 0.5pt solid #ccc;
    padding: 4pt 6pt;
    vertical-align: top;
    font-size: 8.5pt;
}

tbody tr:last-child td {
    border-bottom: 1.5pt solid #333;
}

/* ── Code blocks ── */
pre {
    background-color: #f7f7f7;
    border: 0.5pt solid #ddd;
    border-radius: 2pt;
    padding: 8pt 10pt;
    font-family: "Courier New", "DejaVu Sans Mono", Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.35;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
    margin: 8pt 0 12pt 0;
}

code {
    font-family: "Courier New", "DejaVu Sans Mono", Courier, monospace;
    font-size: 8.5pt;
    background-color: #f3f3f3;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre code {
    background: none;
    padding: 0;
    font-size: 7.5pt;
}

/* ── Lists ── */
ul, ol {
    margin: 4pt 0 10pt 18pt;
    padding-left: 12pt;
}

li {
    margin-bottom: 3pt;
    font-size: 10pt;
}

/* ── Blockquotes (for notes) ── */
blockquote {
    margin: 8pt 0.2in;
    padding: 6pt 10pt;
    border-left: 2.5pt solid #999;
    background: #fafafa;
    font-size: 9.5pt;
    font-style: italic;
    color: #444;
}

/* ── Horizontal rules (section dividers) ── */
hr {
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 16pt 0;
}

/* ── Strong / emphasis ── */
strong {
    font-weight: bold;
}

em {
    font-style: italic;
}

/* ── References section ── */
.references h3 {
    font-size: 10.5pt;
    font-weight: bold;
    margin-top: 12pt;
}

.references ol {
    font-size: 9pt;
    line-height: 1.35;
}

/* ── Appendices ── */
.appendices {
    font-size: 9.5pt;
}

/* ── Figures and diagrams (ASCII art) ── */
.diagram {
    text-align: center;
    margin: 12pt 0;
}
"""


# ─── Markdown transformations for publication ──────────────────────────────

def strip_metadata_header(md_text: str) -> tuple[str, dict]:
    """Extract and clean the metadata header, returning (cleaned_md, metadata)."""
    lines = md_text.split('\n')
    metadata = {}
    title = ''
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('# ') and not title:
            title = stripped[2:].strip()
            metadata['title'] = title
            continue
        if stripped.startswith('**') and ':**' in stripped:
            key_match = re.match(r'\*\*(.+?):\*\*\s*(.*)', stripped)
            if key_match:
                key = key_match.group(1).strip()
                val = key_match.group(2).strip()
                metadata[key] = val
                continue
        if stripped.startswith('>'):
            continue  # Skip blockquote callouts
        if stripped == '---':
            body_start = i + 1
            break
        if stripped == '':
            continue

    return '\n'.join(lines[body_start:]), metadata


def remove_toc(md_text: str) -> str:
    """Remove the Table of Contents section."""
    # Match from "## Table of Contents" to the next "---", any "## " heading,
    # or the end of the document.
    cleaned = re.sub(
        r'## Table of Contents\s*\n(?:.*?\n)*?(?=---|\n## |\Z)',
        '', md_text
    )
    return cleaned


def remove_about_author(md_text: str) -> str:
    """Remove the 'About the Author' section and everything after it."""
    pattern = r'\n## About the Author.*'
    cleaned = re.sub(pattern, '', md_text, flags=re.DOTALL)
    # Also remove trailing footer lines
    cleaned = re.sub(r'\n\*Report generated from.*', '', cleaned, flags=re.DOTALL)
    return cleaned.rstrip() + '\n'


def remove_trailing_footers(md_text: str) -> str:
    """Remove trailing italic footer lines."""
    lines = md_text.rstrip().split('\n')
    while lines and (lines[-1].strip().startswith('*') and lines[-1].strip().endswith('*')):
        lines.pop()
    while lines and lines[-1].strip() == '':
        lines.pop()
    return '\n'.join(lines) + '\n'


def clean_for_publication(md_text: str) -> tuple[str, dict]:
    """Apply all publication transformations to markdown text."""
    body, metadata = strip_metadata_header(md_text)
    body = remove_toc(body)
    body = remove_about_author(body)
    body = remove_trailing_footers(body)
    return body, metadata


# ─── HTML generation ───────────────────────────────────────────────────────

def build_title_block(metadata: dict) -> str:
    """Build an HTML title block from extracted metadata."""
    title = metadata.get('title', 'Untitled')
    author = metadata.get('Author', metadata.get('author', 'Unknown'))
    institution = metadata.get('Institution', '')
    date = metadata.get('Date', '')
    compliance = metadata.get('AI Standards Compliance', '')
    project = metadata.get('Project', '')

    html = '<div class="title-block">\n'
    html += f'  <h1>{title}</h1>\n'
    if project and project != title:
        html += f'  <p style="font-size: 10pt; margin: 2pt 0 8pt 0; color: #444;">{project}</p>\n'
    html += f'  <p class="author">{author}</p>\n'
    if institution:
        html += f'  <p class="affiliation">{institution}</p>\n'
    if date:
        html += f'  <p class="date">{date}</p>\n'
    if compliance:
        html += f'  <p class="compliance">Standards Compliance: {compliance}</p>\n'
    html += '</div>\n'
    return html


def md_to_publication_html(md_text: str) -> str:
    """Convert markdown text to publication-formatted HTML."""
    body, metadata = clean_for_publication(md_text)

    # Split abstract from body
    abstract_html = ''
    keywords_html = ''
    remaining_body = body

    # Extract abstract
    abstract_match = re.search(
        r'## Abstract\s*\n(.*?)(?=\n\*\*Keywords:\*\*|\n---|\n## )',
        body, re.DOTALL
    )
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # Convert markdown bold/italic in abstract to HTML
        abstract_converted = markdown.markdown(abstract_text)
        abstract_html = f'<div class="abstract"><h2>Abstract</h2>{abstract_converted}</div>\n'

    # Extract keywords
    kw_match = re.search(r'\*\*Keywords:\*\*\s*(.*?)(?:\n\n|\n---)', body, re.DOTALL)
    if kw_match:
        kw_text = kw_match.group(1).strip()
        # Convert any markdown formatting in keywords
        kw_text = re.sub(r'\*\*(.+?)\*\*', r'\1', kw_text)
        keywords_html = f'<div class="keywords"><strong>Keywords:</strong> {kw_text}</div>\n'

    # Remove abstract and keywords from body to avoid duplication
    remaining_body = re.sub(
        r'## Abstract\s*\n.*?(?=\n---\s*\n\s*\n*## (?!Abstract))',
        '', body, count=1, flags=re.DOTALL
    )
    # Clean up leading separators
    remaining_body = re.sub(r'^\s*---\s*\n', '', remaining_body)

    # Convert remaining markdown to HTML
    extensions = ['tables', 'fenced_code', 'codehilite', 'sane_lists']
    body_html = markdown.markdown(remaining_body, extensions=extensions)

    # Build full HTML document
    title_block = build_title_block(metadata)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{metadata.get('title', 'Publication')}</title>
</head>
<body>
{title_block}
{abstract_html}
{keywords_html}
<hr>
{body_html}
</body>
</html>"""

    return html


# ─── PDF generation ────────────────────────────────────────────────────────

def generate_pdf(md_path: str, pdf_path: str):
    """Generate a publication-ready PDF from a markdown file."""
    md_text = Path(md_path).read_text(encoding='utf-8')
    html_content = md_to_publication_html(md_text)
    css = CSS(string=PUBLICATION_CSS)
    HTML(string=html_content).write_pdf(pdf_path, stylesheets=[css])
    print(f"  Generated: {pdf_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

REPORTS = [
    {
        'md': 'AI Safety Red-Team Evaluation_ Technical Analysis Report (3).md',
        'pdf': 'AI_Safety_RedTeam_Evaluation_Publication.pdf',
    },
    {
        'md': 'Breast_Cancer_Classification_Report (4).md',
        'pdf': 'Breast_Cancer_Classification_Publication.pdf',
    },
    {
        'md': 'LLM_Ensemble_Bias_Detection_Report (3).md',
        'pdf': 'LLM_Bias_Detection_Publication.pdf',
    },
]


def main():
    base = Path(__file__).parent
    print("Generating publication-ready PDFs...\n")

    for report in REPORTS:
        md_path = base / report['md']
        pdf_path = base / report['pdf']

        if not md_path.exists():
            print(f"  SKIP (not found): {report['md']}")
            continue

        print(f"  Processing: {report['md']}")
        generate_pdf(str(md_path), str(pdf_path))

    print("\nDone. All publication PDFs generated.")


if __name__ == '__main__':
    main()
