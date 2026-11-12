"""Convert report/FINAL_REPORT.md and report/SUPPLEMENT.md to .html (A4, Arial 11 pt, 1.5 line spacing, figures embedded)
and to .docx (python-docx + htmldocx; figures embedded, A4, Arial 11 pt, 1.5 spacing). HTML comments (draft notes) are dropped; [[…]] items
for the applicant are highlighted in yellow so that none is overlooked.
"""
import base64
import re

import markdown
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Cm, Pt
from htmldocx import HtmlToDocx

from common import WORK

R = WORK.parent / "report"

CSS = """@page { size: A4; margin: 2.5cm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; color: #000; background: #fff; }
h1 { font-size: 16pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; } h4 { font-size: 11pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 10pt; }
th, td { border: 1px solid #888; padding: 2pt 5pt; text-align: left; vertical-align: top; }
img { max-width: 16cm; }
.todo { background: #fff3a0; }"""


def normalise_lists(md):
    """python-markdown needs a blank line before a list and 4-space nesting; the report uses GitHub style
    (list directly under a bold line, 2-space nesting)."""
    out, prev = [], ""
    for line in md.split("\n"):
        m = re.match(r"^( *)([-*]|\d+\.) ", line)
        if m:
            line = " " * (len(m.group(1)) // 2 * 4) + line.lstrip(" ")
            if prev.strip() and not re.match(r"^ *([-*]|\d+\.) ", prev):
                out.append("")
        elif prev and re.match(r"^ *([-*]|\d+\.) ", prev) and line.startswith("  ") and line.strip():
            line = " " * 4 + line.lstrip(" ")   # continuation line of a list item
        out.append(line)
        prev = line
    return "\n".join(out)


def build(name, title):
    md = (R / f"{name}.md").read_text()
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    html = markdown.markdown(normalise_lists(md), extensions=["tables", "sane_lists"])
    html = re.sub(r"\[\[(.*?)\]\]", r'<span class="todo">[[\1]]</span>', html)

    # .docx: local figure files; table cells get no markup the converter cannot handle
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(11)
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    for h in ["Heading 1", "Heading 2", "Heading 3", "Heading 4"]:
        d.styles[h].font.name = "Arial"
    conv = HtmlToDocx()
    conv.table_style = "Table Grid"
    conv.add_html_to_document(re.sub(r'src="(figures/[^"]+\.png)"', lambda m: f'src="{R / m.group(1)}"', html), d)
    for sh in d.inline_shapes:   # fit figures to the text width
        r = sh.height / sh.width
        sh.width = Cm(16); sh.height = int(Cm(16) * r)
    d.save(R / f"{name}.docx")

    def embed(m):
        data = base64.b64encode((R / m.group(1)).read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'
    html = re.sub(r'src="(figures/[^"]+\.png)"', embed, html)
    doc = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{html}</body></html>'
    (R / f"{name}.html").write_text(doc)
    print("written", R / f"{name}.html", R / f"{name}.docx", "| open [[…]] items:", len(re.findall(r"\[\[", md)))


def main():
    build("FINAL_REPORT", "Final report")
    build("SUPPLEMENT", "Final report: supplement")


if __name__ == "__main__":
    main()
