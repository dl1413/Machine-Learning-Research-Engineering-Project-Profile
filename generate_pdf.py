"""
Convert Markdown research paper to PDF publication.
"""

import markdown2
from weasyprint import HTML, CSS
from pathlib import Path

def convert_md_to_pdf(md_file, output_pdf):
    """
    Convert markdown file to professional PDF.

    Args:
        md_file (str): Path to markdown file
        output_pdf (str): Path to output PDF
    """
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=['fenced-code-blocks', 'tables', 'header-ids', 'footnotes']
    )

    # Create styled HTML document
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: letter;
                margin: 1in;
                @bottom-center {{
                    content: counter(page);
                }}
            }}
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                font-size: 20pt;
                font-weight: bold;
                margin-top: 24pt;
                margin-bottom: 12pt;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 8pt;
                color: #2c3e50;
            }}
            h2 {{
                font-size: 16pt;
                font-weight: bold;
                margin-top: 20pt;
                margin-bottom: 10pt;
                color: #34495e;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 4pt;
            }}
            h3 {{
                font-size: 13pt;
                font-weight: bold;
                margin-top: 16pt;
                margin-bottom: 8pt;
                color: #34495e;
            }}
            h4 {{
                font-size: 11pt;
                font-weight: bold;
                margin-top: 12pt;
                margin-bottom: 6pt;
                color: #7f8c8d;
            }}
            p {{
                margin-top: 8pt;
                margin-bottom: 8pt;
                text-align: justify;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 16pt 0;
                font-size: 10pt;
            }}
            th {{
                background-color: #34495e;
                color: white;
                font-weight: bold;
                padding: 8pt;
                text-align: left;
                border: 1px solid #2c3e50;
            }}
            td {{
                padding: 6pt 8pt;
                border: 1px solid #bdc3c7;
            }}
            tr:nth-child(even) {{
                background-color: #ecf0f1;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2pt 4pt;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 12pt;
                border-left: 4px solid #3498db;
                overflow-x: auto;
                font-size: 9pt;
                line-height: 1.4;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            ul, ol {{
                margin-top: 8pt;
                margin-bottom: 8pt;
                padding-left: 24pt;
            }}
            li {{
                margin-bottom: 4pt;
            }}
            strong {{
                font-weight: bold;
                color: #2c3e50;
            }}
            em {{
                font-style: italic;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 16pt;
                margin: 16pt 0;
                color: #555;
                font-style: italic;
            }}
            hr {{
                border: none;
                border-top: 2px solid #bdc3c7;
                margin: 24pt 0;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 16pt auto;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    HTML(string=styled_html).write_pdf(output_pdf)

    print(f"✓ PDF generated successfully: {output_pdf}")
    print(f"  Size: {Path(output_pdf).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    md_file = "/home/runner/work/Machine-Learning-Research-Engineering-Project-Profile/Machine-Learning-Research-Engineering-Project-Profile/Aaron_Judge_HomeRun_Projection_Research_Paper.md"
    output_pdf = "/home/runner/work/Machine-Learning-Research-Engineering-Project-Profile/Machine-Learning-Research-Engineering-Project-Profile/Aaron_Judge_HomeRun_Projection_Publication.pdf"

    convert_md_to_pdf(md_file, output_pdf)
