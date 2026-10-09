#!/usr/bin/env python3
"""v4.md -> SIGMOD-style Word document (reviewed Chinese first draft).
Layout: A4; body text in SimSun 10.5pt, justified, first-line indent of 2 characters; headings in SimHei by level;
three-line table style; hanging indent for the references."""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "./reports/paper_sigmod2027_cn_v4.md"
OUT = "./reports/paper_sigmod2027_cn_v4.docx"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin, sec.bottom_margin = Cm(2.54), Cm(2.54)

def set_font(run, name_cn="宋体", name_en="Times New Roman", size=10.5, bold=False):
    run.font.name = name_en
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), name_cn)

def para(text, size=10.5, bold=False, cn="宋体", align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         indent=True, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.15
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    set_font(r, cn, size=size, bold=bold)
    return p

def heading(text, level):
    sizes = {1: 15, 2: 12, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, "黑体", size=sizes.get(level, 11), bold=True)

def add_table(rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            c = t.cell(ri, ci)
            c.text = ""
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(cell)
            set_font(r, "宋体", size=9, bold=(ri == 0))
            if ri == 0:
                sh = OxmlElement("w:shd"); sh.set(qn("w:fill"), "EFEFEF")
                c._tc.get_or_add_tcPr().append(sh)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

lines = open(SRC).read().split("\n")
i = 0
in_refs = False
while i < len(lines):
    line = lines[i].rstrip()
    if not line.strip():
        i += 1; continue
    if line.startswith("# ") and i < 3:
        para(line[2:], size=16, bold=True, cn="黑体",
             align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=4)
    elif line.startswith("## "):
        h = line[3:]
        in_refs = h.startswith("参考文献")
        heading(h, 1)
    elif line.startswith("### "):
        heading(line[4:], 2)
    elif line.startswith("|"):
        tbl = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"-{2,}:?|-{3,}", c) for c in cells):
                tbl.append(cells)
            i += 1
        add_table(tbl)
        continue
    elif re.match(r"^\[\d+\]", line.strip()):
        p = para(line.strip(), size=9, indent=False)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
    elif line.startswith("（") or line.startswith("副标题"):
        para(line, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    else:
        para(line)
    i += 1

doc.save(OUT)
print("saved", OUT)
