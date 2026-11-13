#!/usr/bin/env python
"""Convert docs/Bao_cao_day_du.md -> docs/Bao_cao_day_du.docx (python-docx)."""
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = Path("/home/clyde/DATN/docs/Bao_cao_day_du.md")
DST = Path("/home/clyde/DATN/docs/Bao_cao_day_du.docx")

NAVY = RGBColor(0x0F, 0x2B, 0x4C)
ACCENT = RGBColor(0xB8, 0x86, 0x2D)   # gold accent
GREY = RGBColor(0x59, 0x5F, 0x6B)
BASE_FONT = "Times New Roman"
CODE_FONT = "Consolas"

doc = Document()

# ---------- page setup ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)   # A4
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)

# ---------- styles ----------
styles = doc.styles
normal = styles["Normal"]
normal.font.name = BASE_FONT
normal.font.size = Pt(12)
normal.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
normal.paragraph_format.space_before = Pt(2)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
normal.paragraph_format.line_spacing = 1.18

def style_heading(name, size, color, bold=True, space_before=14, space_after=8, base=True):
    st = styles[name]
    st.font.name = BASE_FONT
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.color.rgb = color
    st.paragraph_format.space_before = Pt(space_before)
    st.paragraph_format.space_after = Pt(space_after)
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.line_spacing = 1.15

style_heading("Title", 24, NAVY, space_before=6, space_after=6)
style_heading("Heading 1", 17, NAVY, space_before=18, space_after=8)
style_heading("Heading 2", 13.5, ACCENT, space_before=12, space_after=5)
style_heading("Heading 3", 12, GREY, bold=True, space_before=8, space_after=4)

# ---------- footer with page number ----------
footer = sec.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run("Deep Hybrid Recommender — Đồ án tốt nghiệp | trang ")
run.font.name = BASE_FONT
run.font.size = Pt(9)
run.font.color.rgb = GREY
# PAGE field
def add_page_field(paragraph):
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = paragraph.add_run()
    r.font.name = BASE_FONT; r.font.size = Pt(9); r.font.color.rgb = GREY
    r._r.append(fld1); r._r.append(it); r._r.append(fld2)
add_page_field(fp)

# ---------- helpers ----------
def set_cell_bg(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)

def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    trPr.append(tblHeader)

def add_field_instr_run(paragraph, instr):
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = paragraph.add_run()
    r._r.append(fld1); r._r.append(it); r._r.append(fld2)

TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")

def add_runs(paragraph, text, base_size=12, base_bold=False, base_color=None):
    parts = TOKEN_RE.split(text)
    for part in parts:
        if not part:
            continue
        bold = base_bold
        font = BASE_FONT
        size = base_size
        color = base_color
        content = part
        if part.startswith("**") and part.endswith("**"):
            bold = True
            content = part[2:-2]
        elif part.startswith("`") and part.endswith("`"):
            font = CODE_FONT
            size = Pt(base_size - 2).pt if hasattr(Pt(base_size), 'pt') else base_size - 2
            content = part[1:-1]
        r = paragraph.add_run(content)
        r.bold = bold
        r.font.name = font
        r.font.size = Pt(size) if isinstance(size, (int, float)) else size
        if color is not None:
            r.font.color.rgb = color

def add_code_block(code_lines):
    for line in code_lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(line if line else " ")
        r.font.name = CODE_FONT
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    # spacing paragraph after block
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4)
    sp.paragraph_format.line_spacing = 1.0

def add_table(rows):
    n_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.paragraphs[0].paragraph_format.space_before = Pt(2)
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.1
            add_runs(cell.paragraphs[0], value, base_size=10.5,
                     base_bold=(i == 0),
                     base_color=NAVY if i == 0 else None)
            if i == 0:
                set_cell_bg(cell, "0F2B4C")
            elif i % 2 == 0:
                set_cell_bg(cell, "F5F1E6")
    if n_cols:
        set_repeat_table_header(table.rows[0])
    # header text white on navy: re-colour runs
    for c in table.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.bold = True
    return table

def split_table_row(line):
    body = line.strip().strip("|")
    return [cell.strip() for cell in body.split("|")]

def is_table_sep(line):
    return bool(re.match(r"^\|?\s*:?-{3,}", line.strip())) and set(line.strip()) <= set("|-: ")

# ---------- parse markdown ----------
lines = SRC.read_text(encoding="utf-8").splitlines()
i = 0
in_code = False
code_buf = []
table_buf = []
n_h1 = 0
first_title_done = False

while i < len(lines):
    line = lines[i]

    if line.strip().startswith("```"):
        if in_code:
            add_code_block(code_buf)
            code_buf, in_code = [], False
        else:
            in_code = True
        i += 1
        continue

    if in_code:
        code_buf.append(line)
        i += 1
        continue

    # flush table buffer on non-table line
    if table_buf and not line.strip().startswith("|"):
        add_table(table_buf)
        table_buf = []

    stripped = line.strip()

    if not stripped:
        i += 1
        continue

    if stripped == "---":
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run("─" * 60)
        r.font.color.rgb = RGBColor(0xD8, 0xD2, 0xC2)
        r.font.size = Pt(8)
        i += 1
        continue

    if stripped.startswith("|"):
        if is_table_sep(stripped):
            i += 1
            continue
        table_buf.append(split_table_row(stripped))
        i += 1
        continue

    m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
    if m:
        level = len(m.group(1))
        text = m.group(2)
        if level == 1:
            p = doc.add_paragraph()
            p.style = doc.styles["Title"]
            add_runs(p, text, base_size=24, base_bold=True, base_color=NAVY)
            first_title_done = True
        elif level == 2:
            p = doc.add_paragraph(style="Heading 1")
            add_runs(p, text, base_size=17, base_bold=True, base_color=NAVY)
        elif level == 3:
            p = doc.add_paragraph(style="Heading 2")
            add_runs(p, text, base_size=13.5, base_bold=True, base_color=ACCENT)
        else:
            p = doc.add_paragraph(style="Heading 3")
            add_runs(p, text, base_size=12, base_bold=True, base_color=GREY)
        i += 1
        continue

    if stripped.startswith(">"):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        add_runs(p, stripped[1:].strip(), base_size=11, base_color=GREY)
        for r in p.runs:
            r.italic = True
        i += 1
        continue

    m = re.match(r"^[-*]\s+(.*)$", stripped)
    if m:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run("•  ")
        r.font.color.rgb = ACCENT
        r.font.size = Pt(12)
        add_runs(p, m.group(1), base_size=12)
        i += 1
        continue

    m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
    if m:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(f"{m.group(1)}.  ")
        r.font.bold = True
        r.font.color.rgb = NAVY
        add_runs(p, m.group(2), base_size=12)
        i += 1
        continue

    p = doc.add_paragraph()
    add_runs(p, stripped)
    i += 1

if table_buf:
    add_table(table_buf)

# ---------- core properties ----------
props = doc.core_properties
props.title = "Deep Hybrid Recommender — Sườn báo cáo (chi tiết, có lập luận & cơ sở lý thuyết)"
props.author = "DATN — Deep Hybrid Recommender"
props.subject = "Đồ án tốt nghiệp"

doc.save(DST)
print("saved", DST)
