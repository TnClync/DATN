#!/usr/bin/env python
"""Convert docs/Bao_cao_day_du.md -> docs/Bao_cao_TDTU.docx using TDTU template format.

Uses template at docs/TemplateBaoCaoMonHoc_KhoaCNTT.dotx.docx as base,
preserving styles (Nidungvnbn, Tiucctrangmu, Bngbiu-nidung, Heading1-3),
headers, footers, page numbering (roman for front matter, arabic for body).
"""
import re
import sys
import os
import zipfile
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
TEMPLATE_DOCX = Path("/tmp/opencode/template.docx")

# Convert .dotx -> .docx
if not TEMPLATE_DOCX.exists():
    os.makedirs("/tmp/opencode", exist_ok=True)
    zin = zipfile.ZipFile(str(TEMPLATE))
    zout = zipfile.ZipFile(str(TEMPLATE_DOCX), 'w', zipfile.ZIP_DEFLATED)
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

def S(name):
    """Get style by name, fallback to Normal."""
    try:
        return doc.styles[name]
    except KeyError:
        return doc.styles["Normal"]

BODY_STYLE = "Nội dung văn bản"
TABLE_STYLE = "Bảng biểu - nội dung"
TITLE_STYLE = "Tiêu đề các trang mở đầu"

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

def add_heading(text, level=1):
    if level == 1:
        p = doc.add_paragraph(text, style=TITLE_STYLE)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif level == 2:
        p = doc.add_paragraph(text, style="Heading 1")
    elif level == 3:
        p = doc.add_paragraph(text, style="Heading 2")
    else:
        p = doc.add_paragraph(text, style="Heading 3")
    return p

def add_para(text, bold=False, italic=False, size=13, align=None, indent=0, color=None):
    p = doc.add_paragraph(style=BODY_STYLE)
    add_runs(p, text, base_size=size, base_bold=bold, base_color=color)
    if italic:
        for r in p.runs:
            r.italic = True
    if align:
        p.alignment = align
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    return p

def add_bullet(text, size=13):
    p = doc.add_paragraph(style=BODY_STYLE)
    p.paragraph_format.left_indent = Cm(0.6)
    add_runs(p, "\u2022  " + text, base_size=size)
    return p

def add_number(num, text, size=13):
    p = doc.add_paragraph(style=BODY_STYLE)
    p.paragraph_format.left_indent = Cm(0.6)
    r = p.add_run(num + ".  ")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    add_runs(p, text, base_size=size)
    return p

def render_table(rows):
    n_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.paragraphs[0].style = S(TABLE_STYLE)
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
    return table

# ---------- Title page ----------
def write_title_page():
    # University name
    p = doc.add_paragraph("TỔNG LIÊN ĐOÀN LAO ĐỘNG VIỆT NAM", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("TRƯỜNG ĐẠI HỌC TÔN ĐỨC THẮNG", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("KHOA CÔNG NGHỆ THÔNG TIN", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    doc.add_paragraph("")

    p = doc.add_paragraph("[HỌ VÀ TÊN SINH VIÊN 1] - [MSSV]", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("[HỌ VÀ TÊN SINH VIÊN 2] - [MSSV]", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    p = doc.add_paragraph("ĐỀ TÀI: DEEP HYBRID RECOMMENDER SYSTEM", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    p = doc.add_paragraph("BÁO CÁO TỐT NGHIỆP", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    p = doc.add_paragraph("Người hướng dẫn: TS. [Tên giảng viên]", style=BODY_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    p = doc.add_paragraph("THÀNH PHỐ HỒ CHÍ MINH, NĂM 2026", style=TITLE_STYLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Page break
    doc.add_page_break()

# ---------- Front matter ----------
def write_front_matter():
    add_heading("LỜI CẢM ƠN", 1)
    add_para("Chúng em xin chân thành cảm ơn ……………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………")
    add_para("TP. Hồ Chí Minh, ngày ... tháng … năm 20..", align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_para("Tác giả", align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_para("(Ký tên và ghi rõ họ tên)", align=WD_ALIGN_PARAGRAPH.RIGHT, size=11)

    add_heading("CAM ĐOAN", 1)
    add_para("Tôi xin cam đoan đây là công trình nghiên cứu của riêng tôi và được sự hướng dẫn khoa học của TS. [Tên giảng viên]. Các nội dung nghiên cứu, kết quả trong đề tài này là trung thực và chính xác. Ngoài ra, trong Dự án còn sử dụng một số nhận xét, đánh giá cũng như số liệu của các tác giả khác, cơ quan tổ chức khác đều có trích dẫn và chú thích nguồn gốc.")
    add_para("Nếu phát hiện có bất kỳ sự gian lận nào tôi xin hoàn toàn chịu trách nhiệm về nội dung Dự án của mình. Trường Đại học Tôn Đức Thắng không liên quan đến những vi phạm tác quyền, bản quyền.")
    add_para("TP. Hồ Chí Minh, ngày … tháng … năm 20..", align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_para("Tác giả", align=WD_ALIGN_PARAGRAPH.RIGHT)
    add_para("(Ký tên và ghi rõ họ tên)", align=WD_ALIGN_PARAGRAPH.RIGHT, size=11)

    add_heading("TÓM TẮT", 1)
    add_para("Đồ án xây dựng hệ thống khuyến nghị sản phẩm lai (Deep Hybrid Recommender) trên dữ liệu Amazon (implicit feedback). Hệ thống kết hợp 4 nhánh tín hiệu (ALS, BPR, LightGCN, SBERT) qua hai kiến trúc fusion: Early Fusion V3 (triển khai chính, Recall@10 = 0,9004) và RGH v4 (kiến trúc nghiên cứu, HR@10 = 0,7983 vs LightGCN 0,7415). Kết quả được kiểm định bằng bootstrap CI95, Wilcoxon signed-rank và McNemar exact theo từng user. Báo cáo nêu rõ giới hạn sampled metric và coverage CF (62,29% user).")

    # TOC field
    add_heading("MỤC LỤC", 1)
    p = doc.add_paragraph()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = p.add_run(); r._r.append(fld1); r._r.append(it); r._r.append(fld2)

    add_heading("DANH MỤC HÌNH VẼ", 1)
    p = doc.add_paragraph()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = 'TOC \\h \\z \\c "Hình"'
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = p.add_run(); r._r.append(fld1); r._r.append(it); r._r.append(fld2)

    add_heading("DANH MỤC BẢNG BIỂU", 1)
    p = doc.add_paragraph()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = 'TOC \\h \\z \\c "Bảng"'
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    r = p.add_run(); r._r.append(fld1); r._r.append(it); r._r.append(fld2)

    add_heading("DANH MỤC CÁC CHỮ VIẾT TẮT", 1)
    abbrevs = [
        ("ALS", "Alternating Least Squares"),
        ("BPR", "Bayesian Personalized Ranking"),
        ("LightGCN", "Light Graph Convolution Network"),
        ("SBERT", "Sentence-BERT"),
        ("BLaIR", "Benchmark LLMs as semantic encoders"),
        ("RGH", "CF-Anchored Residual Gated Hybrid"),
        ("DCN-V2", "Deep & Cross Network V2"),
        ("HR@K", "Hit Rate @ K"),
        ("NDCG@K", "Normalized Discounted Cumulative Gain @ K"),
        ("MRR", "Mean Reciprocal Rank"),
        ("CI95", "Confidence Interval 95%"),
        ("FAISS", "Facebook AI Similarity Search"),
    ]
    table = doc.add_table(rows=len(abbrevs), cols=2)
    table.style = "Table Grid"
    for i, (abbr, full) in enumerate(abbrevs):
        cell0 = table.cell(i, 0); cell1 = table.cell(i, 1)
        cell0.paragraphs[0].style = S(TABLE_STYLE)
        cell1.paragraphs[0].style = S(TABLE_STYLE)
        add_runs(cell0.paragraphs[0], abbr, base_size=11, base_bold=True)
        add_runs(cell1.paragraphs[0], full, base_size=11)

    doc.add_page_break()

# ---------- Parse markdown and write body ----------
def parse_and_write(md_path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
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
                    p = doc.add_paragraph(cl if cl else " ", style=BODY_STYLE)
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
            rows = table_buf
            n_cols = len(rows[0])
            table = doc.add_table(rows=len(rows), cols=n_cols)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    cell = table.cell(ri, ci)
                    cell.paragraphs[0].style = S(TABLE_STYLE)
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
            p = doc.add_paragraph(stripped[1:-1], style=BODY_STYLE)
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
                add_heading(text, 1)
            elif level == 2:
                add_heading(text, 2)
            elif level == 3:
                add_heading(text, 3)
            else:
                add_heading(text, 4)
            i += 1
            continue

        if stripped.startswith(">"):
            p = doc.add_paragraph(stripped[1:].strip(), style=BODY_STYLE)
            for r in p.runs:
                r.italic = True
                r.font.color.rgb = RGBColor(0x59, 0x5F, 0x6B)
            p.paragraph_format.left_indent = Cm(0.6)
            i += 1
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            add_bullet(m.group(1))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            add_number(m.group(1), m.group(2))
            i += 1
            continue

        add_para(stripped)
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
                cell.paragraphs[0].style = S(TABLE_STYLE)
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

# ---------- Main ----------
write_title_page()
write_front_matter()

# Section break: switch to arabic page numbering for body
# (The template already has this; we just continue writing)

# Write report body from markdown
parse_and_write(SRC)

doc.core_properties.title = "Đồ án tốt nghiệp — Deep Hybrid Recommender"
doc.core_properties.author = "DATN — Deep Hybrid Recommender"
doc.core_properties.subject = "Đồ án tốt nghiệp"
doc.save(str(DST))
print("saved", DST)
