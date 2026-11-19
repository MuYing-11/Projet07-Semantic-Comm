"""Petit utilitaire python-pptx : diapositives 16:9 au style homogène (titre, puces, images, tableaux)."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0xC8, 0x7A, 0x1E)
GREY = RGBColor(0x55, 0x5F, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)


class Deck:
    def __init__(self, footer=""):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.footer = footer
        self.n = 0

    # ------------------------------------------------------------------ primitives
    def _text(self, slide, x, y, w, h, text, size=18, bold=False, color=GREY, align=PP_ALIGN.LEFT):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text, p.alignment = text, align
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        return tb

    def _frame(self, title, subtitle=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bar = s.shapes.add_shape(1, 0, 0, self.prs.slide_width, Inches(1.0))
        bar.fill.solid()
        bar.fill.fore_color.rgb = NAVY
        bar.line.fill.background()
        self._text(s, 0.45, 0.16, 12.2, 0.7, title, size=28, bold=True, color=WHITE)
        if subtitle:
            self._text(s, 0.45, 1.05, 12.4, 0.5, subtitle, size=15, color=ACCENT)
        self._text(s, 0.45, 7.05, 10, 0.35, self.footer, size=10, color=GREY)
        self._text(s, 12.2, 7.05, 0.8, 0.35, str(self.n), size=10, color=GREY, align=PP_ALIGN.RIGHT)
        return s

    def _bullets(self, slide, items, x, y, w, h, size=18):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        first = True
        for it in items:
            level = 0
            while it.startswith("  "):
                level, it = level + 1, it[2:]
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = ("• " if level == 0 else "– ") + it
            p.level = level
            p.font.size = Pt(size - 2 * level)
            p.font.color.rgb = GREY if level else RGBColor(0x22, 0x2B, 0x36)
            p.space_after = Pt(6)
        return tb

    # ------------------------------------------------------------------ slide types
    def title(self, title, subtitle, authors, date):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.shapes.add_shape(1, 0, 0, self.prs.slide_width, self.prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()
        self._text(s, 0.8, 2.0, 11.7, 1.6, title, size=38, bold=True, color=WHITE)
        self._text(s, 0.8, 3.7, 11.7, 0.8, subtitle, size=20, color=RGBColor(0xF2, 0xC1, 0x7D))
        self._text(s, 0.8, 5.2, 11.7, 0.9, authors, size=16, color=WHITE)
        self._text(s, 0.8, 6.3, 11.7, 0.5, date, size=14, color=RGBColor(0xC9, 0xD3, 0xE0))

    def section(self, title, subtitle=""):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.shapes.add_shape(1, 0, 0, self.prs.slide_width, self.prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = ACCENT
        bg.line.fill.background()
        self._text(s, 0.8, 2.8, 11.7, 1.2, title, size=36, bold=True, color=WHITE)
        if subtitle:
            self._text(s, 0.8, 4.0, 11.7, 0.8, subtitle, size=18, color=WHITE)

    def bullets(self, title, items, subtitle=None, size=18):
        s = self._frame(title, subtitle)
        self._bullets(s, items, 0.6, 1.6 if subtitle else 1.35, 12.1, 5.4, size)
        return s

    def image(self, title, path, caption=None, subtitle=None, bullets=None, img_width=None):
        s = self._frame(title, subtitle)
        top = 1.55 if subtitle else 1.3
        avail_h = 5.55 - (0.45 if caption else 0)
        if bullets:
            self._bullets(s, bullets, 0.5, top, 4.6, 5.5, size=15)
            left, avail_w = 5.3, 7.6
        else:
            left, avail_w = 0.6, 12.1
        pic = s.shapes.add_picture(path, Inches(left), Inches(top))
        ratio = pic.height / pic.width
        w = min(avail_w if img_width is None else img_width, avail_h / ratio)
        pic.width, pic.height = Inches(w), Inches(w * ratio)
        pic.left = Inches(left + (avail_w - w) / 2)
        if caption:
            self._text(s, left, top + w * ratio + 0.05, avail_w, 0.4, caption, size=12, color=GREY,
                       align=PP_ALIGN.CENTER)
        return s

    def two_images(self, title, left_path, right_path, captions=("", ""), subtitle=None):
        s = self._frame(title, subtitle)
        top = 1.55 if subtitle else 1.35
        for k, path in enumerate((left_path, right_path)):
            pic = s.shapes.add_picture(path, Inches(0.4 + 6.35 * k), Inches(top))
            ratio = pic.height / pic.width
            w = min(6.1, 5.0 / ratio)
            pic.width, pic.height = Inches(w), Inches(w * ratio)
            if captions[k]:
                self._text(s, 0.4 + 6.35 * k, top + w * ratio + 0.05, 6.1, 0.4, captions[k], size=12,
                           align=PP_ALIGN.CENTER)
        return s

    def table(self, title, header, rows, subtitle=None, col_widths=None, size=13, note=None, highlight=None):
        s = self._frame(title, subtitle)
        top = 1.6 if subtitle else 1.4
        n_rows, n_cols = len(rows) + 1, len(header)
        height = min(5.2, 0.42 * n_rows)
        shape = s.shapes.add_table(n_rows, n_cols, Inches(0.5), Inches(top), Inches(12.3), Inches(height))
        tbl = shape.table
        if col_widths:
            tot = sum(col_widths)
            for j, cw in enumerate(col_widths):
                tbl.columns[j].width = Emu(int(Inches(12.3) * cw / tot))
        for j, h in enumerate(header):
            c = tbl.cell(0, j)
            c.text = str(h)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY
            for p in c.text_frame.paragraphs:
                p.font.size, p.font.bold, p.font.color.rgb = Pt(size), True, WHITE
        for i, r in enumerate(rows, 1):
            for j, v in enumerate(r):
                c = tbl.cell(i, j)
                c.text = str(v)
                c.fill.solid()
                c.fill.fore_color.rgb = (RGBColor(0xFF, 0xF1, 0xDB) if highlight and i - 1 in highlight
                                         else (LIGHT if i % 2 else WHITE))
                for p in c.text_frame.paragraphs:
                    p.font.size = Pt(size)
                    p.font.color.rgb = RGBColor(0x22, 0x2B, 0x36)
        if note:
            self._text(s, 0.5, top + height + 0.15, 12.3, 1.2, note, size=13, color=GREY)
        return s

    def save(self, path):
        self.prs.save(path)
