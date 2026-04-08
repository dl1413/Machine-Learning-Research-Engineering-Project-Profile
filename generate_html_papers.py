#!/usr/bin/env python3
"""
Generate NeurIPS/arXiv-formatted HTML documents from project markdown reports.

Produces standalone HTML files with CSS styling that replicates the NeurIPS
conference paper format (matching arXiv:2511.10507v2 style):
  - Single-column, serif typography (Computer Modern / Times)
  - Centered title block with author and affiliation
  - Indented abstract with bold label
  - Numbered sections with booktabs-style tables
  - Numbered references, page numbers at bottom center

Usage:
    python generate_html_papers.py
"""

import re
import markdown
from pathlib import Path


# ─── NeurIPS / arXiv Paper CSS ─────────────────────────────────────────────

NEURIPS_CSS = r"""
/* ─── Import Computer Modern (serif) web font ─── */
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap');

/* ─── Page & Print ─── */
@page {
    size: letter;
    margin: 1in 0.75in 1in 0.75in;
    @bottom-center {
        content: counter(page);
        font-family: "Source Serif 4", "Times New Roman", "DejaVu Serif", Georgia, serif;
        font-size: 10pt;
    }
}
@page :first {
    @bottom-center { content: none; }
}

@media print {
    body { margin: 0; }
    .page-number { display: none; }
}

/* ─── Base ─── */
* { box-sizing: border-box; }

body {
    font-family: "Source Serif 4", "Times New Roman", "DejaVu Serif", Georgia, serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #000;
    max-width: 6.5in;
    margin: 0 auto;
    padding: 40px 20px 60px 20px;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    background: #fff;
}

/* ─── Title Block ─── */
.paper-title {
    text-align: center;
    margin: 0 0 24pt 0;
}

.paper-title h1 {
    font-size: 17pt;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 14pt 0;
    letter-spacing: -0.01em;
}

.paper-title .authors {
    font-size: 12pt;
    font-weight: 600;
    margin: 0 0 4pt 0;
}

.paper-title .affiliation {
    font-size: 10pt;
    font-style: italic;
    color: #333;
    margin: 2pt 0;
}

.paper-title .date {
    font-size: 9pt;
    color: #555;
    margin: 6pt 0 0 0;
}

/* ─── Abstract ─── */
.abstract {
    margin: 20pt 0.4in 16pt 0.4in;
    font-size: 9pt;
    line-height: 1.35;
}

.abstract-label {
    font-weight: 700;
    font-size: 10pt;
    display: block;
    text-align: center;
    margin-bottom: 6pt;
}

.abstract p {
    text-indent: 0;
    margin: 0;
}

/* ─── Keywords ─── */
.keywords {
    margin: 0 0.4in 20pt 0.4in;
    font-size: 8.5pt;
    color: #333;
    line-height: 1.3;
}

.keywords strong {
    color: #000;
}

/* ─── Section Divider ─── */
hr.section-rule {
    border: none;
    border-top: 0.5pt solid #000;
    margin: 20pt 0;
}

/* ─── Section Headings ─── */
h2 {
    font-size: 12pt;
    font-weight: 700;
    margin: 24pt 0 8pt 0;
    padding: 0;
    text-transform: none;
    page-break-after: avoid;
}

h3 {
    font-size: 10.5pt;
    font-weight: 700;
    margin: 16pt 0 6pt 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    font-weight: 700;
    font-style: italic;
    margin: 12pt 0 4pt 0;
    page-break-after: avoid;
}

/* ─── Paragraphs ─── */
p {
    margin: 0 0 6pt 0;
    text-indent: 0;
}

/* ─── Tables (booktabs-style) ─── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 8.5pt;
    line-height: 1.3;
    page-break-inside: avoid;
}

thead th {
    border-top: 2pt solid #000;
    border-bottom: 1pt solid #000;
    padding: 5pt 6pt;
    text-align: left;
    font-weight: 700;
    font-size: 8.5pt;
    background: none;
}

tbody td {
    border-bottom: none;
    padding: 3.5pt 6pt;
    vertical-align: top;
}

tbody tr:last-child td {
    border-bottom: 2pt solid #000;
}

/* Alternating subtle shading for readability */
tbody tr:nth-child(even) {
    background-color: #f9f9f9;
}

caption {
    font-size: 9pt;
    font-weight: 700;
    text-align: left;
    margin-bottom: 4pt;
    caption-side: top;
}

/* ─── Code Blocks ─── */
pre {
    background-color: #f5f5f5;
    border: 0.5pt solid #ddd;
    border-radius: 2pt;
    padding: 8pt 10pt;
    font-family: "Source Code Pro", "Courier New", "DejaVu Sans Mono", monospace;
    font-size: 7.5pt;
    line-height: 1.3;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
    margin: 8pt 0 12pt 0;
}

code {
    font-family: "Source Code Pro", "Courier New", "DejaVu Sans Mono", monospace;
    font-size: 8pt;
    background-color: #f0f0f0;
    padding: 0.5pt 2pt;
    border-radius: 1.5pt;
}

pre code {
    background: none;
    padding: 0;
    font-size: 7.5pt;
}

/* ─── Lists ─── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 22pt;
}

li {
    margin-bottom: 2pt;
}

/* ─── Blockquotes ─── */
blockquote {
    margin: 8pt 18pt;
    padding: 4pt 10pt;
    border-left: 2pt solid #999;
    font-style: italic;
    color: #444;
    font-size: 9pt;
}

/* ─── Horizontal Rules ─── */
hr {
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 16pt 0;
}

/* ─── Strong / Emphasis ─── */
strong { font-weight: 700; }
em { font-style: italic; }

/* ─── References ─── */
.references {
    font-size: 8.5pt;
    line-height: 1.3;
}

.references h3 {
    font-size: 10pt;
    margin-top: 10pt;
}

.references ol {
    padding-left: 18pt;
}

.references li {
    margin-bottom: 3pt;
}

/* ─── Appendices ─── */
.appendices {
    font-size: 9pt;
}

/* ─── Footer / Page Number ─── */
.page-footer {
    text-align: center;
    font-size: 9pt;
    color: #666;
    margin-top: 40pt;
    padding-top: 10pt;
    border-top: 0.5pt solid #ccc;
}

/* ─── Math (basic) ─── */
.math-block {
    text-align: center;
    margin: 10pt 0;
    font-style: italic;
}

/* ─── Responsive (screen reading) ─── */
@media screen and (max-width: 700px) {
    body {
        padding: 20px 15px;
        font-size: 11pt;
    }
    table { font-size: 8pt; }
    pre { font-size: 7pt; }
}
"""


# ─── Markdown transformations (reused from generate_publication_pdfs.py) ────

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
            continue
        if stripped == '---':
            body_start = i + 1
            break
        if stripped == '':
            continue

    return '\n'.join(lines[body_start:]), metadata


def remove_toc(md_text: str) -> str:
    """Remove the Table of Contents section."""
    return re.sub(
        r'## Table of Contents\s*\n(?:.*?\n)*?(?=---|\n## |\Z)',
        '', md_text
    )


def remove_about_author(md_text: str) -> str:
    """Remove the 'About the Author' section and everything after it."""
    cleaned = re.sub(r'\n## About the Author.*', '', md_text, flags=re.DOTALL)
    cleaned = re.sub(r'\n\*Report generated from.*', '', cleaned, flags=re.DOTALL)
    return cleaned.rstrip() + '\n'


def remove_trailing_footers(md_text: str) -> str:
    """Remove trailing italic footer lines."""
    lines = md_text.rstrip().split('\n')
    while lines and lines[-1].strip().startswith('*') and lines[-1].strip().endswith('*'):
        lines.pop()
    while lines and lines[-1].strip() == '':
        lines.pop()
    return '\n'.join(lines) + '\n'


def fix_tables_for_markdown_lib(md_text: str) -> str:
    """Ensure blank line before pipe tables (required by Python markdown lib)."""
    md_text = md_text.replace('Mean |SHAP|', 'Mean SHAP')

    lines = md_text.split('\n')
    result = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code
            result.append(line)
            continue
        if in_code:
            result.append(line)
            continue

        is_table_row = stripped.startswith('|')
        if is_table_row:
            prev = result[-1].strip() if result else ''
            if prev != '' and not prev.startswith('|'):
                result.append('')

        result.append(line)

    return '\n'.join(result)


def clean_for_publication(md_text: str) -> tuple[str, dict]:
    """Apply all publication transformations to markdown text."""
    body, metadata = strip_metadata_header(md_text)
    body = remove_toc(body)
    body = remove_about_author(body)
    body = remove_trailing_footers(body)
    body = fix_tables_for_markdown_lib(body)
    return body, metadata


# ─── HTML generation ───────────────────────────────────────────────────────

def build_neurips_title_block(metadata: dict) -> str:
    """Build a NeurIPS-style title block."""
    title = metadata.get('title', 'Untitled')
    author = metadata.get('Author', 'Unknown')
    institution = metadata.get('Institution', '')
    date = metadata.get('Date', '')

    html = '<div class="paper-title">\n'
    html += f'  <h1>{title}</h1>\n'
    html += f'  <p class="authors">{author}</p>\n'
    if institution:
        html += f'  <p class="affiliation">{institution}</p>\n'
    if date:
        html += f'  <p class="date">{date}</p>\n'
    html += '</div>\n'
    return html


def md_to_neurips_html(md_text: str) -> str:
    """Convert markdown to a NeurIPS-formatted standalone HTML document."""
    body, metadata = clean_for_publication(md_text)

    # ── Extract abstract ──
    abstract_html = ''
    abstract_match = re.search(
        r'## Abstract\s*\n(.*?)(?=\n\*\*Keywords:\*\*|\n---|\n## )',
        body, re.DOTALL
    )
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        abstract_converted = markdown.markdown(abstract_text)
        abstract_html = (
            '<div class="abstract">\n'
            '  <span class="abstract-label">Abstract</span>\n'
            f'  {abstract_converted}\n'
            '</div>\n'
        )

    # ── Extract keywords ──
    keywords_html = ''
    kw_match = re.search(r'\*\*Keywords:\*\*\s*(.*?)(?:\n\n|\n---)', body, re.DOTALL)
    if kw_match:
        kw_text = kw_match.group(1).strip()
        kw_text = re.sub(r'\*\*(.+?)\*\*', r'\1', kw_text)
        keywords_html = (
            f'<div class="keywords"><strong>Keywords:</strong> {kw_text}</div>\n'
        )

    # ── Remove abstract/keywords from body to avoid duplication ──
    remaining_body = re.sub(
        r'## Abstract\s*\n.*?(?=\n---\s*\n\s*\n*## (?!Abstract))',
        '', body, count=1, flags=re.DOTALL
    )
    remaining_body = re.sub(r'^\s*---\s*\n', '', remaining_body)

    # ── Convert remaining markdown to HTML ──
    extensions = ['tables', 'fenced_code', 'codehilite', 'sane_lists']
    body_html = markdown.markdown(remaining_body, extensions=extensions)

    # ── Build full HTML document ──
    title_block = build_neurips_title_block(metadata)
    doc_title = metadata.get('title', 'Publication')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc_title}</title>
    <style>
{NEURIPS_CSS}
    </style>
</head>
<body>

{title_block}
{abstract_html}
{keywords_html}
<hr class="section-rule">

{body_html}

</body>
</html>"""

    return html


# ─── Main ──────────────────────────────────────────────────────────────────

REPORTS = [
    {
        'md': 'AI Safety Red-Team Evaluation_ Technical Analysis Report (3).md',
        'html': 'AI_Safety_RedTeam_Evaluation_Publication.html',
    },
    {
        'md': 'Breast_Cancer_Classification_Report (4).md',
        'html': 'Breast_Cancer_Classification_Publication.html',
    },
    {
        'md': 'LLM_Ensemble_Bias_Detection_Report (3).md',
        'html': 'LLM_Bias_Detection_Publication.html',
    },
]


def main():
    base = Path(__file__).parent
    print("Generating NeurIPS-formatted HTML papers...\n")

    for report in REPORTS:
        md_path = base / report['md']
        html_path = base / report['html']

        if not md_path.exists():
            print(f"  SKIP (not found): {report['md']}")
            continue

        print(f"  Processing: {report['md']}")
        md_text = md_path.read_text(encoding='utf-8')
        html_content = md_to_neurips_html(md_text)
        html_path.write_text(html_content, encoding='utf-8')
        print(f"  Generated:  {html_path.name}")

    print("\nDone. All HTML papers generated.")


if __name__ == '__main__':
    main()
