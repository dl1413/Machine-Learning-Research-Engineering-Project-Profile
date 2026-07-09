#!/usr/bin/env python3
"""
Dependency-free publication PDF renderer (Python stdlib only).

Renders a Markdown technical report to a typeset PDF with embedded fonts,
justified serif body text, styled tables (header shading, zebra striping,
page-break header repeat), monospaced code/diagram blocks, block quotes,
lists, a centered title block, and an abstract box.

Fonts are embedded as CIDFontType2 (Identity-H) so the full Unicode range
used by the report (box-drawing, arrows, ✓/✗, Greek, math symbols) renders
correctly, with per-glyph fallback to DejaVu Sans for any glyph a primary
face lacks.  No third-party packages are required.

This is an offline fallback for `generate_publication_pdfs.py` (which depends
on WeasyPrint) for environments without network access to install packages.
It relies only on TrueType fonts shipped with most Linux distributions
(Liberation Serif, DejaVu Sans/Mono).

Usage:
    python generate_pdf_stdlib.py RAG_Project_Report.md RAG_Project_Publication.pdf
"""

import re
import sys
import zlib
import struct
from pathlib import Path

# ─── Font files ────────────────────────────────────────────────────────────
LIB = "/usr/share/fonts/truetype/liberation"
DJV = "/usr/share/fonts/truetype/dejavu"
FONT_FILES = {
    "serif":            f"{LIB}/LiberationSerif-Regular.ttf",
    "serif-bold":       f"{LIB}/LiberationSerif-Bold.ttf",
    "serif-italic":     f"{LIB}/LiberationSerif-Italic.ttf",
    "serif-bolditalic": f"{LIB}/LiberationSerif-BoldItalic.ttf",
    "mono":             f"{DJV}/DejaVuSansMono.ttf",
    "mono-bold":        f"{DJV}/DejaVuSansMono-Bold.ttf",
    "fallback":         f"{DJV}/DejaVuSans.ttf",
}


# ─── Minimal TrueType parser ───────────────────────────────────────────────
class TTF:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        self._tables = {}
        self._parse_dir()
        self.units = self._head_units()
        self.num_glyphs = self._maxp_numglyphs()
        self.cmap = self._parse_cmap()          # unicode cp -> gid
        self.adv = self._parse_hmtx()           # gid -> advance (font units)
        self.used = set([0])                    # gids we actually emit

    def _u16(self, o): return struct.unpack(">H", self.data[o:o + 2])[0]
    def _s16(self, o): return struct.unpack(">h", self.data[o:o + 2])[0]
    def _u32(self, o): return struct.unpack(">I", self.data[o:o + 4])[0]

    def _parse_dir(self):
        num = self._u16(4)
        o = 12
        for _ in range(num):
            tag = self.data[o:o + 4].decode("latin-1")
            off = self._u32(o + 8)
            ln = self._u32(o + 12)
            self._tables[tag] = (off, ln)
            o += 16

    def _head_units(self):
        o = self._tables["head"][0]
        return self._u16(o + 18)

    def _maxp_numglyphs(self):
        o = self._tables["maxp"][0]
        return self._u16(o + 4)

    def _parse_hmtx(self):
        hhea = self._tables["hhea"][0]
        num_hm = self._u16(hhea + 34)
        hmtx = self._tables["hmtx"][0]
        adv = []
        last = 0
        for i in range(self.num_glyphs):
            if i < num_hm:
                last = self._u16(hmtx + i * 4)
            adv.append(last)
        return adv

    def _parse_cmap(self):
        base = self._tables["cmap"][0]
        ntab = self._u16(base + 2)
        best = None
        for i in range(ntab):
            pid = self._u16(base + 4 + i * 8)
            eid = self._u16(base + 6 + i * 8)
            off = self._u32(base + 8 + i * 8)
            score = {(3, 10): 5, (3, 1): 4, (0, 3): 4, (0, 4): 3, (3, 0): 1}.get((pid, eid), 0)
            if score and (best is None or score > best[0]):
                best = (score, base + off)
        if best is None:
            best = (0, base + self._u32(base + 8))
        return self._parse_cmap_sub(best[1])

    def _parse_cmap_sub(self, o):
        fmt = self._u16(o)
        m = {}
        if fmt == 4:
            segx2 = self._u16(o + 6)
            segc = segx2 // 2
            end_o = o + 14
            start_o = end_o + segx2 + 2
            delta_o = start_o + segx2
            range_o = delta_o + segx2
            for s in range(segc):
                end = self._u16(end_o + s * 2)
                start = self._u16(start_o + s * 2)
                delta = self._u16(delta_o + s * 2)
                rng = self._u16(range_o + s * 2)
                for c in range(start, end + 1):
                    if c == 0xFFFF:
                        continue
                    if rng == 0:
                        g = (c + delta) & 0xFFFF
                    else:
                        gi = range_o + s * 2 + rng + (c - start) * 2
                        g = self._u16(gi)
                        if g != 0:
                            g = (g + delta) & 0xFFFF
                    if g != 0:
                        m[c] = g
        elif fmt == 12:
            ngroups = self._u32(o + 12)
            g_o = o + 16
            for i in range(ngroups):
                sc = self._u32(g_o + i * 12)
                ec = self._u32(g_o + i * 12 + 4)
                sg = self._u32(g_o + i * 12 + 8)
                for c in range(sc, ec + 1):
                    m[c] = sg + (c - sc)
        return m

    def has(self, cp):
        return cp in self.cmap

    def gid(self, cp):
        g = self.cmap.get(cp, 0)
        self.used.add(g)
        return g

    def width1000(self, cp):
        g = self.cmap.get(cp, 0)
        return self.adv[g] * 1000.0 / self.units


# ─── Font manager with per-glyph fallback ──────────────────────────────────
class Fonts:
    def __init__(self):
        self.ttf = {}

    def get(self, key):
        if key not in self.ttf:
            self.ttf[key] = TTF(FONT_FILES[key])
        return self.ttf[key]

    def resolve(self, cp, style):
        """Return font key whose face contains cp (with fallbacks)."""
        chain = {
            "reg":         ["serif", "fallback", "mono"],
            "bold":        ["serif-bold", "fallback", "mono-bold"],
            "italic":      ["serif-italic", "fallback", "mono"],
            "bolditalic":  ["serif-bolditalic", "serif-bold", "fallback"],
            "code":        ["mono", "fallback"],
            "code-bold":   ["mono-bold", "mono", "fallback"],
        }[style]
        for k in chain:
            if self.get(k).has(cp):
                return k
        return chain[0]

    def char_width(self, cp, style, size):
        key = self.resolve(cp, style)
        return self.get(key).width1000(cp) * size / 1000.0


# ─── Inline markdown → styled runs ─────────────────────────────────────────
def parse_inline(text, base_style="reg"):
    """Return list of (char, style). Handles **bold**, *italic*, `code`."""
    out = []
    i = 0
    bold = base_style in ("bold", "bolditalic")
    ital = base_style in ("italic", "bolditalic")
    n = len(text)

    def cur_style(code=False):
        if code:
            return "code-bold" if bold else "code"
        if bold and ital:
            return "bolditalic"
        if bold:
            return "bold"
        if ital:
            return "italic"
        return "reg"

    while i < n:
        c = text[i]
        if c == "`":
            j = text.find("`", i + 1)
            if j == -1:
                out.append((c, cur_style()))
                i += 1
                continue
            for ch in text[i + 1:j]:
                out.append((ch, cur_style(code=True)))
            i = j + 1
            continue
        if text.startswith("**", i):
            bold = not bold
            i += 2
            continue
        if c in "*_":
            # treat single * or _ as italic toggle (avoid ** already handled)
            ital = not ital
            i += 1
            continue
        out.append((c, cur_style()))
        i += 1
    return out


# ─── PDF writer ────────────────────────────────────────────────────────────
class PDF:
    def __init__(self, fonts):
        self.fonts = fonts
        self.objs = {}          # num -> bytes
        self.n = 0
        self.pages = []         # list of content byte-strings
        self.page_font_keys = set()

    def alloc(self):
        self.n += 1
        return self.n

    def add(self, num, data):
        self.objs[num] = data

    # -- font embedding --
    def _font_objects(self):
        """Create PDF objects for each used font; return {key: obj_num}."""
        mapping = {}
        for key, ttf in self.fonts.ttf.items():
            if len(ttf.used) <= 1:
                continue
            mapping[key] = self._embed_one(key, ttf)
        return mapping

    def _embed_one(self, key, ttf):
        type0 = self.alloc()
        cidfont = self.alloc()
        descr = self.alloc()
        fontfile = self.alloc()
        tounicode = self.alloc()

        comp = zlib.compress(ttf.data)
        self.add(fontfile,
                 b"<< /Length %d /Length1 %d /Filter /FlateDecode >>\nstream\n"
                 % (len(comp), len(ttf.data)) + comp + b"\nendstream")

        # widths
        used = sorted(g for g in ttf.used)
        w_parts = []
        for g in used:
            w = round(ttf.adv[g] * 1000.0 / ttf.units)
            w_parts.append(b"%d [%d]" % (g, w))
        warr = b" ".join(w_parts)

        # font bbox / metrics (coarse but valid)
        head = ttf._tables["head"][0]
        xmin = ttf._s16(head + 36) * 1000 // ttf.units
        ymin = ttf._s16(head + 38) * 1000 // ttf.units
        xmax = ttf._s16(head + 40) * 1000 // ttf.units
        ymax = ttf._s16(head + 42) * 1000 // ttf.units
        psname = ("Font" + key.replace("-", "")).encode()

        self.add(descr,
                 b"<< /Type /FontDescriptor /FontName /%s /Flags 4 "
                 b"/FontBBox [%d %d %d %d] /ItalicAngle 0 /Ascent %d /Descent %d "
                 b"/CapHeight %d /StemV 80 /FontFile2 %d 0 R >>"
                 % (psname, xmin, ymin, xmax, ymax, ymax, ymin, ymax, fontfile))

        self.add(cidfont,
                 b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s "
                 b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
                 b"/FontDescriptor %d 0 R /CIDToGIDMap /Identity /W [ %s ] >>"
                 % (psname, descr, warr))

        # ToUnicode
        cmap_entries = []
        rev = {}
        for cp, g in ttf.cmap.items():
            if g in ttf.used and g not in rev:
                rev[g] = cp
        for g in used:
            cp = rev.get(g)
            if cp is not None and cp <= 0xFFFF:
                cmap_entries.append(b"<%04X> <%04X>" % (g, cp))
        tu = (b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
              b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
              b"/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
              b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n")
        for i in range(0, len(cmap_entries), 100):
            chunk = cmap_entries[i:i + 100]
            tu += b"%d beginbfchar\n" % len(chunk) + b"\n".join(chunk) + b"\nendbfchar\n"
        tu += b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
        tuc = zlib.compress(tu)
        self.add(tounicode,
                 b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(tuc)
                 + tuc + b"\nendstream")

        self.add(type0,
                 b"<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H "
                 b"/DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>"
                 % (psname, cidfont, tounicode))
        return type0

    def build(self, layout_pages):
        """layout_pages: list of content-stream byte strings."""
        font_objs = self._font_objects()
        # font resource dict shared by all pages
        res_font = b" ".join(
            b"/F_%s %d 0 R" % (k.replace("-", "_").encode(), num)
            for k, num in font_objs.items()
        )
        pages_obj = self.alloc()
        page_nums = []
        for content in layout_pages:
            c = zlib.compress(content)
            cobj = self.alloc()
            self.add(cobj,
                     b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(c)
                     + c + b"\nendstream")
            pobj = self.alloc()
            self.add(pobj,
                     b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                     b"/Resources << /Font << %s >> >> /Contents %d 0 R >>"
                     % (pages_obj, res_font, cobj))
            page_nums.append(pobj)
        kids = b" ".join(b"%d 0 R" % p for p in page_nums)
        self.add(pages_obj,
                 b"<< /Type /Pages /Count %d /Kids [%s] >>"
                 % (len(page_nums), kids))
        catalog = self.alloc()
        self.add(catalog, b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj)
        info = self.alloc()
        self.add(info,
                 b"<< /Title (RAG Production Pipeline \\226 Technical Report) "
                 b"/Author (Derek Lankeaux) /Creator (stdlib md2pdf) >>")
        return self._serialize(catalog, info)

    def _serialize(self, catalog, info):
        out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for num in range(1, self.n + 1):
            offsets[num] = len(out)
            out += b"%d 0 obj\n" % num + self.objs[num] + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n" % (self.n + 1)
        out += b"0000000000 65535 f \n"
        for num in range(1, self.n + 1):
            out += b"%010d 00000 n \n" % offsets[num]
        out += (b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF"
                % (self.n + 1, catalog, info, xref))
        return bytes(out)


# ─── Layout engine ─────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 612.0, 792.0
ML, MR = 61.2, 61.2
MT, MB = 72.0, 64.0
CW = PAGE_W - ML - MR          # content width
TOP = PAGE_H - MT              # start y
BOTTOM = MB


def esc_none(_):
    return b""


class Layout:
    def __init__(self, fonts):
        self.fonts = fonts
        self.pages = []
        self.buf = []
        self.y = TOP
        self.page_no = 1

    # -- low level --
    def _new_page(self):
        self._flush_footer()
        self.pages.append(b"".join(self.buf))
        self.buf = []
        self.y = TOP
        self.page_no += 1

    def _flush_footer(self):
        if self.page_no == 1:
            return
        s = str(self.page_no)
        size = 9.5
        w = sum(self.fonts.char_width(ord(c), "reg", size) for c in s)
        x = (PAGE_W - w) / 2
        self._raw_text(x, MB - 22, s, "reg", size, (0.2, 0.2, 0.2))

    def ensure(self, h):
        if self.y - h < BOTTOM:
            self._new_page()

    def _gids_hex(self, text, style):
        """Group text into same-font runs -> list of (font_key, hexbytes, width)."""
        runs = []
        cur_key = None
        cur = []
        for ch in text:
            cp = ord(ch)
            key = self.fonts.resolve(cp, style)
            if key != cur_key and cur:
                runs.append((cur_key, cur))
                cur = []
            cur_key = key
            cur.append(cp)
        if cur:
            runs.append((cur_key, cur))
        return runs

    def _raw_text(self, x, y, text, style, size, color=(0, 0, 0)):
        """Draw a plain (single-style) string at absolute x,y."""
        for key, cps in self._gids_hex(text, style):
            ttf = self.fonts.get(key)
            hexs = b"".join(b"%04X" % ttf.gid(cp) for cp in cps)
            w = sum(ttf.width1000(cp) for cp in cps) * size / 1000.0
            self.buf.append(
                b"q %.3f %.3f %.3f rg BT /F_%s %.2f Tf 1 0 0 1 %.2f %.2f Tm <%s> Tj ET Q\n"
                % (color[0], color[1], color[2],
                   key.replace("-", "_").encode(), size, x, y, hexs))
            x += w
        return x

    def _rect(self, x, y, w, h, fill):
        self.buf.append(b"q %.3f %.3f %.3f rg %.2f %.2f %.2f %.2f re f Q\n"
                        % (fill[0], fill[1], fill[2], x, y, w, h))

    def _line(self, x1, y1, x2, y2, width, color=(0.4, 0.4, 0.4)):
        self.buf.append(b"q %.3f %.3f %.3f RG %.2f w %.2f %.2f m %.2f %.2f l S Q\n"
                        % (color[0], color[1], color[2], width, x1, y1, x2, y2))

    # -- styled-run helpers --
    def measure_run(self, styled, size):
        return sum(self.fonts.char_width(ord(c), st, size) for c, st in styled)

    def draw_styled_line(self, x, y, words, size, justify=False, avail=CW):
        """words: list of word-run-lists; each word is [(char,style),...]."""
        space_w = self.fonts.char_width(ord(" "), "reg", size)
        n = len(words)
        if n == 0:
            return
        text_w = sum(self.measure_run(w, size) for w in words)
        gaps = n - 1
        if justify and gaps > 0:
            gap = (avail - text_w) / gaps
            gap = min(gap, space_w * 4)  # avoid rivers on short lines
            if gap < space_w:
                gap = space_w
        else:
            gap = space_w
        cx = x
        for wi, word in enumerate(words):
            # group contiguous same-style within the word too (style may vary)
            run_key = None
            run_cps = []

            def flush(rk, cps, cx):
                if not cps:
                    return cx
                ttf = self.fonts.get(rk)
                hexs = b"".join(b"%04X" % ttf.gid(cp) for cp in cps)
                w = sum(ttf.width1000(cp) for cp in cps) * size / 1000.0
                self.buf.append(
                    b"q 0 0 0 rg BT /F_%s %.2f Tf 1 0 0 1 %.2f %.2f Tm <%s> Tj ET Q\n"
                    % (rk.replace("-", "_").encode(), size, cx, self.y_cur, hexs))
                return cx + w

            self.y_cur = y
            for ch, st in word:
                cp = ord(ch)
                key = self.fonts.resolve(cp, st)
                if key != run_key and run_cps:
                    cx = flush(run_key, run_cps, cx)
                    run_cps = []
                run_key = key
                run_cps.append(cp)
            cx = flush(run_key, run_cps, cx)
            if wi != n - 1:
                cx += gap

    def wrap_words(self, styled, size, avail):
        """Split styled char list into words; greedily wrap into lines."""
        # split into words on spaces
        words = []
        cur = []
        for ch, st in styled:
            if ch == " ":
                if cur:
                    words.append(cur)
                    cur = []
            elif ch == "\n":
                if cur:
                    words.append(cur)
                    cur = []
            else:
                cur.append((ch, st))
        if cur:
            words.append(cur)

        space_w = self.fonts.char_width(ord(" "), "reg", size)
        lines = []
        line = []
        lw = 0.0
        for word in words:
            ww = self.measure_run(word, size)
            add = ww + (space_w if line else 0)
            if line and lw + add > avail:
                lines.append(line)
                line = [word]
                lw = ww
            else:
                line.append(word)
                lw += add
        if line:
            lines.append(line)
        return lines


# ─── Markdown block model ──────────────────────────────────────────────────
def strip_metadata_header(md_text):
    lines = md_text.split("\n")
    meta = {}
    title = ""
    body_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("# ") and not title:
            title = s[2:].strip()
            meta["title"] = title
            continue
        m = re.match(r"\*\*(.+?):\*\*\s*(.*)", s)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
            continue
        if s.startswith(">"):
            continue
        if s == "---":
            body_start = i + 1
            break
        if s == "":
            continue
    return "\n".join(lines[body_start:]), meta


def clean_body(md):
    md = re.sub(r"## Table of Contents\s*\n(?:.*?\n)*?(?=---|\n## [^T])", "", md)
    md = re.sub(r"\n## About the Author.*", "", md, flags=re.DOTALL)
    md = re.sub(r"\n\*\*End of .*?\*\*\s*$", "", md, flags=re.DOTALL)
    return md.rstrip() + "\n"


def extract_abstract_keywords(body):
    ab,kw = "", ""
    m = re.search(r"## Abstract\s*\n(.*?)(?=\n\*\*Keywords:\*\*|\n## )", body, re.DOTALL)
    if m:
        ab = m.group(1).strip()
    m2 = re.search(r"\*\*Keywords:\*\*\s*(.*?)(?:\n\n|\n---|\Z)", body, re.DOTALL)
    if m2:
        kw = re.sub(r"\*\*(.+?)\*\*", r"\1", m2.group(1).strip())
    # remove abstract+keywords block from body
    body = re.sub(r"## Abstract\s*\n.*?(?=\n---\s*\n)", "", body, count=1, flags=re.DOTALL)
    body = re.sub(r"^\s*---\s*\n", "", body)
    return body, ab, kw


TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def parse_blocks(body):
    lines = body.split("\n")
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", code))
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            blocks.append(("math", s[2:-2].strip()))
            i += 1
            continue
        if s.startswith("$$"):
            i += 1
            math = []
            while i < n and not lines[i].strip().startswith("$$"):
                math.append(lines[i])
                i += 1
            i += 1
            blocks.append(("math", " ".join(x.strip() for x in math)))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", s)
        if m:
            blocks.append(("h%d" % len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if s == "---":
            blocks.append(("hr", ""))
            i += 1
            continue
        # table: a line with | and next line separator
        if "|" in line and i + 1 < n and TABLE_SEP.match(lines[i + 1]):
            header = line
            i += 2
            rows = [header]
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(lines[i])
                i += 1
            blocks.append(("table", rows))
            continue
        if s.startswith(">"):
            quote = [re.sub(r"^\s*>\s?", "", line)]
            i += 1
            while i < n and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            blocks.append(("quote", " ".join(quote)))
            continue
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", line)
        if m:
            items = []
            while i < n:
                mm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", lines[i])
                if not mm:
                    if lines[i].strip() == "":
                        break
                    # continuation line
                    if items:
                        items[-1] = (items[-1][0], items[-1][1] + " " + lines[i].strip())
                        i += 1
                        continue
                    break
                ordered = bool(re.match(r"\d+\.", mm.group(2)))
                marker = mm.group(2) if ordered else "•"
                items.append((marker, mm.group(3).strip()))
                i += 1
            blocks.append(("list", items))
            continue
        if s == "":
            i += 1
            continue
        # paragraph: gather consecutive non-empty, non-special lines
        para = [line]
        i += 1
        while i < n:
            nx = lines[i]
            ns = nx.strip()
            if (ns == "" or ns.startswith("#") or ns.startswith("```")
                    or ns.startswith(">") or ns == "---"
                    or ("|" in nx and i + 1 < n and TABLE_SEP.match(lines[i + 1]))
                    or re.match(r"^(\s*)([-*+]|\d+\.)\s+", nx)
                    or ns.startswith("$$")):
                break
            para.append(nx)
            i += 1
        blocks.append(("para", " ".join(x.strip() for x in para)))
    return blocks


def split_table(rows):
    def cells(r):
        r = r.strip()
        if r.startswith("|"):
            r = r[1:]
        if r.endswith("|"):
            r = r[:-1]
        return [c.strip() for c in r.split("|")]
    return [cells(r) for r in rows]


# ─── Renderer: blocks -> layout ────────────────────────────────────────────
GREY = (0.45, 0.45, 0.45)
LGREY = (0.6, 0.6, 0.6)


def render(md_text, out_path):
    fonts = Fonts()
    L = Layout(fonts)

    body, meta = strip_metadata_header(md_text)
    body = clean_body(body)
    body, abstract, keywords = extract_abstract_keywords(body)

    # ---- Title block ----
    def center(text, style, size, color=(0, 0, 0), gap_after=0):
        styled = parse_inline(text, style)
        w = L.measure_run(styled, size)
        x = ML + (CW - w) / 2
        L.ensure(size + 4)
        # draw as one justify=False line
        words = L.wrap_words(styled, size, CW)[0] if styled else []
        # simpler: raw draw run by run centered
        cx = x
        for ch, st in styled:
            cp = ord(ch)
            cx = L._raw_text(cx, L.y, ch, st, size, color)
        L.y -= size + gap_after

    title = meta.get("title", "Report")
    L.y -= 6
    # wrap title if long
    tstyled = parse_inline(title, "bold")
    for ln in L.wrap_words(tstyled, 17, CW):
        w = sum(L.measure_run(word, 17) for word in ln) + (len(ln) - 1) * fonts.char_width(ord(" "), "reg", 17)
        x = ML + (CW - w) / 2
        L.y_cur = L.y
        L.draw_styled_line(x, L.y, ln, 17, justify=False)
        L.y -= 21
    L.y -= 4
    if meta.get("Project"):
        center(meta["Project"], "italic", 10.5, (0.27, 0.27, 0.27), 4)
    center(meta.get("Author", "Author"), "bold", 11.5, (0, 0, 0), 2)
    if meta.get("Role"):
        center(meta["Role"], "reg", 10, (0.2, 0.2, 0.2), 1)
    if meta.get("Institution"):
        center(meta["Institution"], "italic", 10, (0.2, 0.2, 0.2), 1)
    if meta.get("Date"):
        center(meta["Date"], "reg", 10, (0.33, 0.33, 0.33), 2)
    if meta.get("AI Standards Compliance"):
        center("Standards Compliance: " + meta["AI Standards Compliance"], "reg", 8.5, (0.4, 0.4, 0.4), 2)
    L.y -= 4
    L._line(ML, L.y, PAGE_W - MR, L.y, 0.75, (0.2, 0.2, 0.2))
    L.y -= 16

    # ---- Abstract ----
    if abstract:
        center("Abstract", "bold", 11, (0, 0, 0), 4)
        L._line(ML + 20, L.y + 6, PAGE_W - MR - 20, L.y + 6, 0.4, LGREY)
        draw_paragraph(L, fonts, abstract, 9.5, indent=22, justify=True)
        L.y -= 2
        L._line(ML + 20, L.y + 6, PAGE_W - MR - 20, L.y + 6, 0.4, LGREY)
        L.y -= 10
    if keywords:
        draw_kv(L, fonts, "Keywords: ", keywords, 9)
        L.y -= 6
    L._line(ML, L.y, PAGE_W - MR, L.y, 0.4, (0.8, 0.8, 0.8))
    L.y -= 14

    # ---- Body blocks ----
    blocks = parse_blocks(body)
    for kind, payload in blocks:
        if kind == "para":
            draw_paragraph(L, fonts, payload, 10.5, justify=True)
            L.y -= 6
        elif kind == "hr":
            L.ensure(16)
            L.y -= 6
            L._line(ML, L.y, PAGE_W - MR, L.y, 0.4, (0.8, 0.8, 0.8))
            L.y -= 10
        elif re.fullmatch(r"h[1-6]", kind):
            draw_heading(L, fonts, int(kind[1:]), payload)
        elif kind == "table":
            draw_table(L, fonts, split_table(payload))
            L.y -= 8
        elif kind == "code":
            draw_code(L, fonts, payload)
            L.y -= 8
        elif kind == "list":
            draw_list(L, fonts, payload)
            L.y -= 6
        elif kind == "quote":
            draw_quote(L, fonts, payload)
            L.y -= 6
        elif kind == "math":
            draw_math(L, fonts, payload)
            L.y -= 6

    L._new_page()  # flush last
    L.pages.append(b"")  # placeholder removed below
    L.pages.pop()

    pdf = PDF(fonts)
    data = pdf.build(L.pages)
    Path(out_path).write_bytes(data)
    return len(L.pages), len(data)


def draw_paragraph(L, fonts, text, size, indent=0, justify=True, color=(0, 0, 0)):
    styled = parse_inline(text, "reg")
    avail = CW - indent
    lines = L.wrap_words(styled, size, avail)
    lh = size * 1.42
    for idx, ln in enumerate(lines):
        L.ensure(lh)
        L.y_cur = L.y
        last = idx == len(lines) - 1
        L.draw_styled_line(ML + indent, L.y, ln, size,
                           justify=(justify and not last), avail=avail)
        L.y -= lh


def draw_kv(L, fonts, key, val, size):
    styled = parse_inline("**" + key + "**" + val, "reg")
    avail = CW - 22
    lines = L.wrap_words(styled, size, avail)
    lh = size * 1.4
    for ln in lines:
        L.ensure(lh)
        L.y_cur = L.y
        L.draw_styled_line(ML + 20, L.y, ln, size, justify=False, avail=avail)
        L.y -= lh


def draw_heading(L, fonts, level, text):
    sizes = {1: 16, 2: 13, 3: 11.5, 4: 10.5}
    size = sizes.get(level, 10.5)
    top = {1: 20, 2: 20, 3: 14, 4: 11}.get(level, 10)
    style = "bolditalic" if level >= 4 else "bold"
    L.ensure(size + top + 10)
    L.y -= top
    styled = parse_inline(text, style)
    lines = L.wrap_words(styled, size, CW)
    for ln in lines:
        L.y_cur = L.y
        L.draw_styled_line(ML, L.y, ln, size, justify=False)
        L.y -= size * 1.25
    if level == 2:
        L.y += 2
        L._line(ML, L.y, PAGE_W - MR, L.y, 0.5, (0.6, 0.6, 0.6))
        L.y -= 8
    else:
        L.y -= 3


def draw_list(L, fonts, items):
    size = 10
    lh = size * 1.4
    for marker, text in items:
        styled = parse_inline(text, "reg")
        avail = CW - 26
        lines = L.wrap_words(styled, size, avail)
        for idx, ln in enumerate(lines):
            L.ensure(lh)
            L.y_cur = L.y
            if idx == 0:
                L._raw_text(ML + 8, L.y, marker, "reg", size)
            L.draw_styled_line(ML + 24, L.y, ln, size, justify=False, avail=avail)
            L.y -= lh
        L.y -= 1


def draw_quote(L, fonts, text):
    size = 9.5
    styled = parse_inline(text, "italic")
    avail = CW - 30
    lines = L.wrap_words(styled, size, avail)
    lh = size * 1.4
    h = lh * len(lines) + 8
    L.ensure(h)
    y0 = L.y + 2
    L._rect(ML + 6, L.y - lh * len(lines) + 2, CW - 6, lh * len(lines) + 2, (0.98, 0.98, 0.98))
    L._line(ML + 6, L.y + 2, ML + 6, L.y - lh * len(lines) + 4, 2.5, (0.6, 0.6, 0.6))
    for ln in lines:
        L.y_cur = L.y
        L.draw_styled_line(ML + 16, L.y, ln, size, justify=False, avail=avail)
        L.y -= lh


def draw_math(L, fonts, text):
    size = 10
    styled = parse_inline(text, "italic")
    w = L.measure_run(styled, size)
    L.ensure(size * 1.6)
    x = ML + max(0, (CW - w) / 2)
    L.y_cur = L.y
    L.draw_styled_line(x, L.y, [[(c, st) for c, st in styled]], size, justify=False)
    L.y -= size * 1.6


def draw_code(L, fonts, lines):
    size = 7.5
    lh = size * 1.32
    pad = 6
    style = "code"
    # split overly long lines (rare) by char budget
    maxchars = int((CW - 2 * pad) / (fonts.char_width(ord("M"), "code", size)))
    flat = []
    for ln in lines:
        ln = ln.rstrip("\n")
        if len(ln) <= maxchars:
            flat.append(ln)
        else:
            while len(ln) > maxchars:
                flat.append(ln[:maxchars])
                ln = ln[maxchars:]
            flat.append(ln)
    # render with page breaks
    idx = 0
    while idx < len(flat):
        # how many lines fit on this page?
        avail_h = L.y - BOTTOM - 2 * pad
        can = max(1, int(avail_h / lh))
        chunk = flat[idx:idx + can]
        if not chunk:
            L._new_page()
            continue
        block_h = lh * len(chunk) + 2 * pad
        if L.y - block_h < BOTTOM and idx == 0 and L.y < TOP - 1:
            L._new_page()
            continue
        top = L.y
        L._rect(ML, top - block_h, CW, block_h, (0.968, 0.968, 0.968))
        L._line(ML, top, ML + CW, top, 0.5, (0.87, 0.87, 0.87))
        L._line(ML, top - block_h, ML + CW, top - block_h, 0.5, (0.87, 0.87, 0.87))
        ty = top - pad - size
        for ln in chunk:
            L._raw_text(ML + pad, ty, ln if ln else " ", style, size, (0.1, 0.1, 0.1))
            ty -= lh
        L.y = top - block_h
        idx += can
        if idx < len(flat):
            L._new_page()


def draw_table(L, fonts, rows):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:]
    hsize, csize = 8.5, 8.5
    pad = 5.0
    # natural widths
    nat = [0.0] * ncol
    for j in range(ncol):
        st = "bold"
        w = L.measure_run(parse_inline(header[j], "bold"), hsize) + 2 * pad
        nat[j] = w
    for r in body:
        for j in range(ncol):
            w = L.measure_run(parse_inline(r[j], "reg"), csize) + 2 * pad
            nat[j] = max(nat[j], min(w, CW * 0.5))
    total = sum(nat)
    if total <= CW:
        # expand proportionally to fill width
        scale = CW / total
        colw = [w * scale for w in nat]
    else:
        scale = CW / total
        colw = [w * scale for w in nat]

    def cell_lines(text, style, w):
        styled = parse_inline(text, style)
        return L.wrap_words(styled, csize, w - 2 * pad) or [[]]

    def row_height(r, style):
        lines = [cell_lines(r[j], style, colw[j]) for j in range(ncol)]
        maxln = max(len(x) for x in lines)
        return maxln * (csize * 1.3) + 2 * pad, lines

    def draw_row(r, style, fill, top, is_header, last=False):
        h, lines = row_height(r, "bold" if is_header else style)
        if fill:
            L._rect(ML, top - h, CW, h, fill)
        # borders
        if is_header:
            L._line(ML, top, ML + CW, top, 1.5, (0.2, 0.2, 0.2))
            L._line(ML, top - h, ML + CW, top - h, 1.0, (0.2, 0.2, 0.2))
        else:
            bw = 1.5 if last else 0.5
            bc = (0.2, 0.2, 0.2) if last else (0.85, 0.85, 0.85)
            L._line(ML, top - h, ML + CW, top - h, bw, bc)
        x = ML
        for j in range(ncol):
            ty = top - pad - csize
            for ln in lines[j]:
                L.y_cur = ty
                L.draw_styled_line(x + pad, ty, ln, csize,
                                   justify=False, avail=colw[j] - 2 * pad)
                ty -= csize * 1.3
            x += colw[j]
        return h

    # header (with repeat on page breaks)
    hh, _ = row_height(header, "bold")
    L.ensure(hh + 20)
    top = L.y
    top -= draw_row(header, "bold", (0.925, 0.925, 0.925), top, True)
    for ri, r in enumerate(body):
        rh, _ = row_height(r, "reg")
        if top - rh < BOTTOM:
            L.y = top
            L._new_page()
            top = L.y
            top -= draw_row(header, "bold", (0.925, 0.925, 0.925), top, True)
        fill = (0.965, 0.965, 0.965) if ri % 2 == 1 else None
        last = ri == len(body) - 1
        top -= draw_row(r, "reg", fill, top, False, last=last)
    L.y = top


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2]
    md = Path(src).read_text(encoding="utf-8")
    pages, size = render(md, out)
    print(f"Wrote {out}: {pages} pages, {size/1024:.1f} KB")
