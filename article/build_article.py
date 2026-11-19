"""Builds the scientific article from a single content source:
    article/article.docx   (Word, figures embedded, equations rendered with matplotlib mathtext)
    article/main.tex       (IEEEtran two-column source, compile on Overleaf with the figures/ folder)
All numbers are read from results/tables/summary.csv (python scripts/make_figures.py first).

    python article/build_article.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from content import AUTHORS, AFFILIATION, KEYWORDS, TITLE, build_blocks, REFERENCES  # noqa: E402

OUT = os.path.join(ROOT, "article")
EQ_DIR = os.path.join(OUT, "eq")


# ------------------------------------------------------------------------------------------------ LaTeX
TEX_MAP = {"π": r"$\pi$", "α": r"$\alpha$", "β": r"$\beta$", "Δ": r"$\Delta$", "γ": r"$\gamma$", "λ": r"$\lambda$",
           "θ": r"$\theta$", "ε": r"$\epsilon$", "σ": r"$\sigma$", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$",
           "×": r"$\times$", "→": r"$\rightarrow$", "≈": r"$\approx$", "·": r"$\cdot$", "₂": r"$_2$", "−": "--",
           "–": "--", "—": "---", "ō": r"$\bar{o}$", "∈": r"$\in$", "∑": r"$\sum$", "’": "'", "²": r"$^2$",
           "°": r"$^\circ$", "“": "``", "”": "''"}


def tex(s):
    out = []
    for part in re.split(r"(\$[^$]+\$)", s):  # keep inline math untouched
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(part)
            continue
        part = part.replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
            part = part.replace(a, b)
        part = re.sub(r"\[(\d+(?:[,–-]\s?\d+)*)\]", lambda m: r"\cite{" + ",".join(
            f"r{k}" for k in _expand(m.group(1))) + "}", part)
        for a, b in TEX_MAP.items():
            part = part.replace(a, b)
        part = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", part)
        part = re.sub(r"\*(.+?)\*", r"\\emph{\1}", part)
        out.append(part)
    return "".join(out)


def _expand(s):
    ks = []
    for tok in re.split(r",\s?", s):
        if re.match(r"\d+[–-]\d+", tok):
            a, b = map(int, re.split(r"[–-]", tok))
            ks += list(range(a, b + 1))
        else:
            ks.append(int(tok))
    return ks


def write_tex(blocks, abstract):
    L = [r"\documentclass[conference]{IEEEtran}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
         r"\usepackage{amsmath,amssymb,graphicx,booktabs,cite}", r"\usepackage[hidelinks]{hyperref}",
         r"\graphicspath{{../figures/}}", r"\begin{document}", rf"\title{{{tex(TITLE)}}}",
         r"\author{" + r" \and ".join(rf"\IEEEauthorblockN{{{tex(a)}}}" for a in AUTHORS) +
         rf"\\ \IEEEauthorblockA{{{tex(AFFILIATION)}}}}}", r"\maketitle",
         r"\begin{abstract}" + tex(abstract) + r"\end{abstract}",
         r"\begin{IEEEkeywords}" + tex(", ".join(KEYWORDS)) + r"\end{IEEEkeywords}"]
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            L.append(rf"\section{{{tex(b[1])}}}")
        elif kind == "h2":
            L.append(rf"\subsection{{{tex(b[1])}}}")
        elif kind == "p":
            L.append(tex(b[1]) + "\n")
        elif kind == "bullets":
            L.append(r"\begin{itemize}" + "".join(r"\item " + tex(i) for i in b[1]) + r"\end{itemize}")
        elif kind == "eq":
            L.append(r"\begin{equation}" + b[1] + r"\end{equation}")
        elif kind == "fig":
            _, path, caption, wide = b
            env = "figure*" if wide else "figure"
            width = r"\textwidth" if wide else r"\columnwidth"
            L.append(rf"\begin{{{env}}}[t]\centering\includegraphics[width={width}]{{{os.path.basename(path)}}}"
                     rf"\caption{{{tex(caption)}}}\end{{{env}}}")
        elif kind == "table":
            _, caption, header, rows, wide = b
            env = "table*" if wide else "table"
            cols = "l" + "c" * (len(header) - 1)
            body = " \\\\\n".join(" & ".join(tex(str(c)) for c in r) for r in rows)
            L.append(rf"\begin{{{env}}}[t]\centering\caption{{{tex(caption)}}}\footnotesize"
                     rf"\begin{{tabular}}{{{cols}}}\toprule " + " & ".join(tex(h) for h in header) +
                     r" \\ \midrule " + body + r" \\ \bottomrule\end{tabular}" + rf"\end{{{env}}}")
    L.append(r"\begin{thebibliography}{99}")
    for k, ref in enumerate(REFERENCES, 1):
        L.append(rf"\bibitem{{r{k}}} {tex(ref)}")
    L += [r"\end{thebibliography}", r"\end{document}"]
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ Word
def render_eq(latex, idx):
    os.makedirs(EQ_DIR, exist_ok=True)
    path = os.path.join(EQ_DIR, f"eq{idx}.png")
    expr = latex.replace(r"\,", " ").replace(r"\left", "").replace(r"\right", "").replace(r"\text", r"\mathrm")
    expr = expr.replace(r"\operatorname", r"\mathrm").replace(r"\big", "").replace(r"\Big", "")
    fig = plt.figure(figsize=(6, 0.5))
    fig.text(0.5, 0.5, f"${expr}$", ha="center", va="center", fontsize=13)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)
    return path


def _font(run, size=10, bold=False, italic=False, color=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size, run.bold, run.italic = Pt(size), bold, italic
    if color:
        run.font.color.rgb = color


def _rich(par, text, size=10):
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|\$[^$]+\$)", text):
        if not part:
            continue
        if part.startswith("**"):
            _font(par.add_run(part[2:-2]), size, bold=True)
        elif part.startswith("*") and len(part) > 2:
            _font(par.add_run(part[1:-1]), size, italic=True)
        elif part.startswith("$"):
            plain = re.sub(r"\\(mathrm|text|operatorname)\{([^}]*)\}", r"\2", part[1:-1])
            for a, b in ((r"\varphi", "φ"), (r"\hat{x}", "x̂"), (r"\eta", "η"), (r"\sigma", "σ"),
                         (r"\mathbb{C}", "ℂ"), (r"\mathbb{R}", "ℝ"), (r"\mathcal{CN}", "CN"), (r"\times", "×"),
                         (r"\|", "‖"), (r"\frac{1}{k}", "(1/k)"), (r"\sum", "Σ"), (r"\sqrt", "√"),
                         (r"\ldots", "…"), (r"\pi", "π"), (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"), (r"\lambda", "λ"),
                         (r"\theta", "θ"), (r"\Delta", "Δ"), (r"\in", "∈"), (r"\le", "≤"), (r"\ge", "≥"),
                         (r"\cdot", "·"), (r"\mathcal{N}", "N"), (r"\bar{o}", "ō"), ("{", ""), ("}", ""), ("\\", "")):
                plain = plain.replace(a, b)
            _font(par.add_run(plain), size, italic=True)
        else:
            _font(par.add_run(part), size)


def _two_columns(section, n=2):
    cols = section._sectPr.xpath("./w:cols")
    col = cols[0] if cols else OxmlElement("w:cols")
    col.set(qn("w:num"), str(n))
    col.set(qn("w:space"), "360")
    if not cols:
        section._sectPr.append(col)


def write_docx(blocks, abstract):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin, sec.bottom_margin = Cm(1.9), Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(TITLE), 20)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(", ".join(AUTHORS)), 11)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(p.add_run(AFFILIATION), 9.5, italic=True)
    body = doc.add_section(WD_SECTION.CONTINUOUS)
    _two_columns(body)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _font(p.add_run("Abstract—"), 9, bold=True, italic=True)
    _rich(p, abstract, 9)
    for r in p.runs[1:]:
        r.bold = True
    p = doc.add_paragraph()
    _font(p.add_run("Index Terms—"), 9, bold=True, italic=True)
    _font(p.add_run(", ".join(KEYWORDS)), 9, bold=True)
    h1 = h2 = fig = tab = eq = 0
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            h2 = 0
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            _font(p.add_run(f"{roman[h1]}. {b[1].upper()}"), 10)
            h1 += 1
        elif kind == "h2":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            _font(p.add_run(f"{'ABCDEFGH'[h2]}. {b[1]}"), 10, italic=True)
            h2 += 1
        elif kind == "p":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Cm(0.35)
            p.paragraph_format.space_after = Pt(2)
            _rich(p, b[1])
        elif kind == "bullets":
            for it in b[1]:
                p = doc.add_paragraph(style="List Bullet")
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                _rich(p, it)
        elif kind == "eq":
            eq += 1
            tbl = doc.add_table(rows=1, cols=2)
            tbl.autofit = False
            c0, c1 = tbl.rows[0].cells
            c0.width, c1.width = Cm(7.2), Cm(0.9)
            c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            c0.paragraphs[0].add_run().add_picture(render_eq(b[1], eq), height=Cm(0.62))
            c1.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _font(c1.paragraphs[0].add_run(f"({eq})"), 10)
        elif kind == "fig":
            fig += 1
            _, path, caption, wide = b
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(path, width=Cm(8.3))
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _font(p.add_run(f"Fig. {fig}. "), 8)
            _rich(p, caption, 8)
        elif kind == "table":
            tab += 1
            _, caption, header, rows, wide = b
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _font(p.add_run(f"TABLE {roman[tab - 1]}\n"), 8)
            _font(p.add_run(caption.upper()), 8)
            t = doc.add_table(rows=len(rows) + 1, cols=len(header))
            t.style = "Table Grid"
            for j, h in enumerate(header):
                cell = t.cell(0, j)
                cell.text = ""
                _font(cell.paragraphs[0].add_run(str(h)), 7, bold=True)
            for i, r in enumerate(rows, 1):
                for j, v in enumerate(r):
                    cell = t.cell(i, j)
                    cell.text = ""
                    _font(cell.paragraphs[0].add_run(str(v)), 7)
            doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    _font(p.add_run("REFERENCES"), 10)
    for k, ref in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        _font(p.add_run(f"[{k}] "), 8)
        _rich(p, ref, 8)
    doc.save(os.path.join(OUT, "article.docx"))


def main():
    t = pd.read_csv(os.path.join("results", "tables", "summary.csv"))
    abstract, blocks = build_blocks(t)
    n_words = len(abstract.split())
    write_docx(blocks, abstract)
    write_tex(blocks, abstract)
    print(f"abstract: {n_words} words; wrote article/article.docx and article/main.tex")


if __name__ == "__main__":
    main()
