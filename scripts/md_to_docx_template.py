#!/usr/bin/env python
"""Convert docs/Bao_cao_day_du.md -> docs/Bao_cao_TDTU.docx using TDTU template format.

Uses the template at docs/TemplateBaoCaoMonHoc_KhoaCNTT.dotx.docx as base,
preserving styles, headers, footers, page numbering.
"""
import re
import sys
import zipfile
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/Bao_cao_day_du.md")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/Bao_cao_TDTU.docx")
TEMPLATE = Path("docs/TemplateBaoCaoMonHoc_KhoaCNTT.dotx.docx")

# Convert template to docx if needed
TEMPLATE_DOCX = Path("/tmp/opencode/template.docx")
if not TEMPLATE_DOCX.exists():
    import os
    os.makedirs("/tmp/opencode", exist_ok=True)
    import zipfile as zf
    zin = zf.ZipFile(str(TEMPLATE))
    zout = zf.ZipFile(str(TEMPLATE_DOCX), 'w', zf.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "[Content_Types].xml":
            data = data.decode("utf-8").replace(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
            ).encode("utf-8")
        zout.writestr(item, data)
    zout.close()
    zin.close()

doc = Document(str(TEMPLATE_DOCX))

# Clear body content (keep sectPr)
body = doc.element.body
sectPr = body.find(qn('w:sectPr'))
for child in list(body):
    if child is not sectPr and child.tag != qn('w:sectPr'):
        body.remove(child)

# ---------- helpers ----------
def add_page_field(paragraph):
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = paragraph.add_run()
    r.font.size = Pt(10)
    r._r.append(fld1); r._r.append(it); r._r.append(fld2)

def add_runs(paragraph, text, base_size=None, base_bold=False, base_color=None, font="Times New Roman"):
    TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    parts = TOKEN_RE.split(text)
    for part in parts:
        if not part:
            continue
        bold = base_bold
        content = part
        if part.startswith("**") and part.endswith("**"):
            bold = True
            content = part[2:-2]
        elif part.startswith("`") and part.endswith("`"):
            content = part[1:-1]
        r = paragraph.add_run(content)
        r.bold = bold
        r.font.name = font
        if base_size:
            r.font.size = Pt(base_size)
        if base_color is not None:
            r.font.color.rgb = base_color

def add_image(path, width_cm=16.0):
    from pathlib import Path
    p = Path(path)
    if not p.is_absolute():
        p = SRC.parent / p
    if p.exists():
        pic_para = doc.add_paragraph()
        pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_para.paragraph_format.space_before = Pt(6)
        pic_para.paragraph_format.space_after = Pt(2)
        run = pic_para.add_run()
        run.add_picture(str(p), width=Cm(width_cm))
    else:
        print("WARNING: missing image", p)

# ---------- parse markdown ----------
lines = SRC.read_text(encoding="utf-8").splitlines()
i = 0
in_code = False
code_buf = []
table_buf = []

while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if stripped.startswith("```"):
        if in_code:
            for cl in code_buf:
                p = doc.add_paragraph(cl if cl else " ", style="Nidungvnbn")
                for r in p.runs:
                    r.font.name = "Consolas"; r.font.size = Pt(9)
            code_buf = []; in_code = False
        else:
            in_code = True
        i += 1
        continue

    if in_code:
        code_buf.append(line)
        i += 1
        continue

    if table_buf and not stripped.startswith("|"):
        # render table
        rows = table_buf
        n_cols = len(rows[0])
        table = doc.add_table(rows=len(rows), cols=n_cols)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = table.cell(ri, ci)
                cell.paragraphs[0].style = doc.styles["Bảng biểu - nội dung"] if "Bảng biểu - nội dung" in [s.name for s in doc.styles] else doc.styles["Normal"]
                add_runs(cell.paragraphs[0], val, base_size=11, base_bold=(ri == 0))
                if ri == 0:
                    for r in cell.paragraphs[0].runs:
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        r.bold = True
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "0F2B4C")
                    tcPr.append(shd)
                elif ri % 2 == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F5F1E6")
                    tcPr.append(shd)
        table_buf = []

    if not stripped:
        i += 1
        continue

    if stripped.startswith("<!--"):
        i += 1
        continue

    if stripped == "---":
        i += 1
        continue

    if stripped.startswith("|"):
        if re.match(r"^\|?\s*:?-{3,}", stripped):
            i += 1
            continue
        table_buf.append([c.strip() for c in stripped.strip("|").split("|")])
        i += 1
        continue

    # images
    m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", stripped)
    if m:
        add_image(m.group(1), width_cm=16.0)
        i += 1
        continue

    # captions
    if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**") and len(stripped) > 3:
        p = doc.add_paragraph(stripped[1:-1], style="Nidungvnbn")
        for r in p.runs:
            r.italic = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x59, 0x5F, 0x6B)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        i += 1
        continue

    # headings
    m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
    if m:
        level = len(m.group(1))
        text = m.group(2)
        if level == 1:
            # Chapter title style
            p = doc.add_paragraph(text, style="Tiucctrangmu")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif level == 2:
            p = doc.add_paragraph(text, style="Heading1")
        elif level == 3:
            p = doc.add_paragraph(text, style="Heading2")
        else:
            p = doc.add_paragraph(text, style="Heading3")
        i += 1
        continue

    if stripped.startswith(">"):
        p = doc.add_paragraph(stripped[1:].strip(), style="Nidungvnbn")
        for r in p.runs:
            r.italic = True
            r.font.color.rgb = RGBColor(0x59, 0x5F, 0x6B)
        p.paragraph_format.left_indent = Cm(0.6)
        i += 1
        continue

    # bullet list
    m = re.match(r"^[-*]\s+(.*)$", stripped)
    if m:
        p = doc.add_paragraph(style="Nidungvnbn")
        p.paragraph_format.left_indent = Cm(0.6)
        add_runs(p, "•  " + m.group(1), base_size=13)
        i += 1
        continue

    # numbered list
    m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
    if m:
        p = doc.add_paragraph(style="Nidungvnbn")
        p.paragraph_format.left_indent = Cm(0.6)
        add_runs(p, m.group(1) + ".  " + m.group(2), base_size=13, base_bold=False)
        i += 1
        continue

    # normal paragraph
    p = doc.add_paragraph(style="Nidungvnbn")
    add_runs(p, stripped, base_size=13)
    i += 1

if table_buf:
    rows = table_buf
    n_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.paragraphs[0].style = doc.styles["Bảng biểu - nội dung"] if "Bảng biểu - nội dung" in [s.name for s in doc.styles] else doc.styles["Normal"]
            add_runs(cell.paragraphs[0], val, base_size=11, base_bold=(ri == 0))
            if ri == 0:
                for r in cell.paragraphs[0].runs:
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    r.bold = True
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "0F2B4C")
                tcPr.append(shd)
            elif ri % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F5F1E6")
                tcPr.append(shd)
    table_buf = []

doc.core_properties.title = "Đồ án tốt nghiệp — Deep Hybrid Recommender"
doc.core_properties.author = "DATN — Deep Hybrid Recommender"
doc.save(str(DST))
print("saved", DST)
