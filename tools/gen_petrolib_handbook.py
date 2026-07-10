#!/usr/bin/env python3
"""Generate doc/Petrophysics_Handbook.pdf — the petrolib user manual.

Layout (fixed by request):

* title page carrying exactly three lines — "Petrophysics Handbook",
  "Code for Petrophysics Jan/Feb 2014 - Jun/Jul 2026", and
  "https://github.com/ingehap/Petrophysics_Code";
* one blank page;
* an alphabetical table of contents of every public petrolib function;
* one page per function: Python path, purpose, input parameters (each with
  its meaning), output parameters, and the sources as full SPWLA
  *Petrophysics* journal citations resolved from the module References
  sections.

Parameter meanings and output descriptions live in
tools/handbook_descriptions.json (keyed module -> function -> params /
returns); keep it in sync when the petrolib API changes — missing entries
are reported on stderr.  Requires reportlab:

    pip install reportlab
    python tools/gen_petrolib_handbook.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "Petrophysics_Handbook.pdf"
DESCRIPTIONS = pathlib.Path(__file__).with_name("handbook_descriptions.json")

RETURN_ALIASES = {
    "_Float": "numpy.ndarray of float64",
    "_Bool": "numpy.ndarray of bool",
    "_Int": "numpy.ndarray of int",
    "_Complex": "numpy.ndarray of complex128",
}


def pretty_annotation(ann: str) -> str:
    """Expand the house type aliases inside a (possibly compound) annotation."""
    if ann in RETURN_ALIASES:
        return RETURN_ALIASES[ann]
    for alias, expanded in (
        ("_Float", "ndarray[float64]"),
        ("_Bool", "ndarray[bool]"),
        ("_Int", "ndarray[int]"),
        ("_Complex", "ndarray[complex128]"),
    ):
        ann = ann.replace(alias, expanded)
    return ann


# ------------------------------------------------------------- extraction


@dataclass
class Param:
    name: str
    annotation: str
    default: str | None
    kw_only: bool
    meaning: str = ""


@dataclass
class Entry:
    name: str
    module: str
    purpose: str
    params: list[Param] = field(default_factory=list)
    returns: str = ""
    returns_meaning: str = ""
    sources: list[str] = field(default_factory=list)


def parse_references(module_doc: str) -> dict[str, str]:
    """Map srcYYYY_MM tags to full citations from a References section."""
    refs: dict[str, str] = {}
    lines = module_doc.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "References")
    except StopIteration:
        return refs
    tag = None
    for ln in lines[start + 2 :]:
        m = re.match(r"(src20\d\d_\d\d\S*) -- (.*)", ln.strip())
        if m:
            tag = m.group(1)
            refs[tag] = m.group(2)
        elif tag and ln.startswith("  ") and ln.strip():
            refs[tag] += " " + ln.strip()
        elif tag and not ln.strip():
            tag = None
    return refs


def split_purpose_sources(doc: str) -> tuple[str, list[str]]:
    """Return (purpose prose, src tags) from a function docstring.

    Tags are taken from the ``Sources:`` clause only, so incidental
    src-tag mentions in the prose (e.g. "the src2018_10/src2018_02
    convention") do not leak into the citation list.
    """
    # rejoin tags wrapped across lines ("src2024_10/\n    ml_permeability")
    joined = re.sub(r"(src20\d\d_\d\d/)\s*\n\s*", r"\1", doc)
    tags: list[str] = []
    for m in re.finditer(r"Sources?:(.*?)(?=\n\n|\Z)", joined, flags=re.S):
        for tok in re.findall(r"src20\d\d_\d\d(?:/[A-Za-z0-9_]+)?", m.group(1)):
            if tok not in tags:
                tags.append(tok)
    purpose = re.sub(r"Sources?:.*?(?=\n\n|\Z)", "", joined, flags=re.S)
    paragraphs = [" ".join(seg.split()) for seg in purpose.split("\n\n") if seg.strip()]
    return "\n\n".join(paragraphs), tags


def resolve_source(tag: str, refs: dict[str, str]) -> str | None:
    """Full citation for a src tag; tolerate a missing /article suffix."""
    if tag in refs:
        return refs[tag]
    hits = [k for k in refs if k.startswith(tag + "/")]
    if len(hits) == 1:
        return refs[hits[0]]
    return None


def params_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Param]:
    out: list[Param] = []
    a = node.args
    pos = a.posonlyargs + a.args
    defaults: list[ast.expr | None] = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults, strict=True):
        if arg.arg in ("self", "cls"):
            continue
        out.append(
            Param(
                arg.arg,
                ast.unparse(arg.annotation) if arg.annotation else "",
                ast.unparse(d) if d is not None else None,
                kw_only=False,
            )
        )
    if a.vararg:
        out.append(Param("*" + a.vararg.arg, "", None, kw_only=False))
    for arg, kd in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append(
            Param(
                arg.arg,
                ast.unparse(arg.annotation) if arg.annotation else "",
                ast.unparse(kd) if kd is not None else None,
                kw_only=True,
            )
        )
    return out


def collect_entries() -> list[Entry]:
    descriptions = json.loads(DESCRIPTIONS.read_text()) if DESCRIPTIONS.exists() else {}
    missing: list[str] = []
    entries: list[Entry] = []
    files = sorted(
        p
        for p in ROOT.glob("petrolib/**/*.py")
        if "__pycache__" not in str(p) and p.name not in ("__init__.py", "_compat.py")
    )
    for pfile in files:
        rel = pfile.relative_to(ROOT)
        modname = "petrolib." + str(rel.with_suffix("")).removeprefix("petrolib/").replace("/", ".")
        tree = ast.parse(pfile.read_text())
        refs = parse_references(ast.get_docstring(tree) or "")
        mod_desc = descriptions.get(modname, {})
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            purpose, tags = split_purpose_sources(doc)
            params = params_of(node)
            fdesc = mod_desc.get(node.name, {})
            for p in params:
                p.meaning = fdesc.get("params", {}).get(p.name, "")
                if not p.meaning:
                    missing.append(f"{modname}.{node.name}({p.name})")
            ret = ast.unparse(node.returns) if node.returns else ""
            sources = []
            for t in tags:
                cite = resolve_source(t, refs)
                if cite is None:
                    missing.append(f"{modname}.{node.name} citation {t}")
                    cite = t
                sources.append(cite)
            entries.append(
                Entry(
                    name=node.name,
                    module=modname,
                    purpose=purpose,
                    params=params,
                    returns=pretty_annotation(ret),
                    returns_meaning=fdesc.get("returns", ""),
                    sources=sources,
                )
            )
    if missing:
        print(f"WARNING: {len(missing)} unresolved descriptions/citations:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
    entries.sort(key=lambda e: (e.name.lower(), e.name, e.module))
    return entries


# ------------------------------------------------------------- rendering


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(text: str) -> str:
    """Docstring prose -> platypus markup (``code`` and :role:`target`)."""
    text = esc(text)
    text = re.sub(r":\w+:`~?([^`]+)`", r"\1", text)
    text = re.sub(r"``([^`]+)``", r'<font face="DejaVuSansMono" size="8.5">\1</font>', text)
    return text


def build_pdf(entries: list[Entry]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        KeepInFrame,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    fdir = pathlib.Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(fdir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(fdir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(fdir / "DejaVuSansMono.ttf")))
    # DejaVu ships no Sans-Oblique on this system; Liberation Sans Italic stands in
    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSans-Oblique",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        )
    )

    navy = colors.Color(0.10, 0.20, 0.36)
    grey = colors.Color(0.35, 0.35, 0.40)
    body = ParagraphStyle("Body", fontName="DejaVuSans", fontSize=9.2, leading=12.4, spaceAfter=4)
    label = ParagraphStyle(
        "Label",
        parent=body,
        fontName="DejaVuSans-Bold",
        fontSize=9.6,
        textColor=navy,
        spaceBefore=0,
        spaceAfter=3,
    )
    func_heading = ParagraphStyle(
        "FuncHeading",
        fontName="DejaVuSansMono",
        fontSize=13,
        leading=16,
        spaceAfter=0,
        textColor=navy,
    )
    src_style = ParagraphStyle(
        "Src", parent=body, fontSize=8.6, leading=11.4, leftIndent=10, spaceAfter=3
    )

    class Handbook(BaseDocTemplate):
        def afterFlowable(self, flowable: object) -> None:
            if isinstance(flowable, Paragraph) and flowable.style.name == "FuncHeading":
                text = getattr(flowable, "_toc_text", flowable.getPlainText())
                key = getattr(flowable, "_toc_key", None)
                self.notify("TOCEntry", (0, text, self.page, key))

    doc = Handbook(
        str(OUT),
        pagesize=A4,
        title="Petrophysics Handbook",
        author="Petrophysics_Code / petrolib",
        subject="User manual for the petrolib common petrophysics library",
    )
    frame_h = A4[1] - 40 * mm
    frame_w = A4[0] - 44 * mm
    frame = Frame(22 * mm, 20 * mm, frame_w, frame_h, id="main")

    def on_page(canvas: object, _doc: object) -> None:
        canvas.saveState()
        canvas.setFont("DejaVuSans", 8)
        canvas.setFillColor(grey)
        canvas.drawString(22 * mm, A4[1] - 13 * mm, "Petrophysics Handbook")
        canvas.drawRightString(A4[0] - 22 * mm, A4[1] - 13 * mm, "petrolib reference")
        canvas.drawCentredString(A4[0] / 2, 11 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.addPageTemplates(
        [
            # 'front' pages (title, blank) carry no header/footer text at all
            PageTemplate(id="front", frames=[frame]),
            PageTemplate(id="normal", frames=[frame], onPage=on_page),
        ]
    )

    story: list[object] = []

    # ---- title page: exactly three lines of text, nothing else
    story.append(Spacer(1, 80 * mm))
    story.append(
        Paragraph(
            "Petrophysics Handbook",
            ParagraphStyle(
                "Title",
                fontName="DejaVuSans-Bold",
                fontSize=34,
                leading=40,
                alignment=1,
            ),
        )
    )
    story.append(Spacer(1, 14 * mm))
    subtitle = ParagraphStyle(
        "Sub", fontName="DejaVuSans", fontSize=14, leading=19, alignment=1, textColor=grey
    )
    story.append(Paragraph("Code for Petrophysics Jan/Feb 2014 - Jun/Jul 2026", subtitle))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("https://github.com/ingehap/Petrophysics_Code", subtitle))

    # ---- blank page
    story.append(NextPageTemplate("front"))
    story.append(PageBreak())
    story.append(Spacer(1, 1))
    story.append(NextPageTemplate("normal"))
    story.append(PageBreak())

    # ---- table of contents
    story.append(
        Paragraph(
            "Table of Contents",
            ParagraphStyle(
                "TocTitle",
                fontName="DejaVuSans-Bold",
                fontSize=18,
                textColor=navy,
                spaceAfter=10,
            ),
        )
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TocEntry",
            fontName="DejaVuSans",
            fontSize=8.2,
            leading=10.6,
            leftIndent=2,
        )
    ]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())

    # ---- one page per function
    blank_line = Spacer(1, 12.4)  # one body-height line with no text
    for i, e in enumerate(entries):
        key = f"e{i}"
        path = f"{e.module}.{e.name}"
        head = Paragraph(f'<a name="{key}"/>{esc(path)}', func_heading)
        head._toc_key = key
        head._toc_text = f"{e.name}  —  {e.module}"
        story.append(head)

        block: list[object] = [blank_line]
        block.append(Paragraph("Purpose", label))
        if e.purpose:
            for para in e.purpose.split("\n\n"):
                block.append(Paragraph(rich(para), body))
        else:
            block.append(Paragraph("(No docstring.)", body))

        block.append(blank_line)
        block.append(Paragraph("Input parameters", label))
        if e.params:
            rows = [["Parameter", "Type", "Default", "Meaning"]]
            for p in e.params:
                rows.append(
                    [
                        Paragraph(
                            f'<font face="DejaVuSansMono" size="8.2">{esc(p.name)}</font>'
                            + (" <i>(kw)</i>" if p.kw_only else ""),
                            body,
                        ),
                        Paragraph(esc(pretty_annotation(p.annotation)) or "—", body),
                        Paragraph(
                            esc(p.default) if p.default is not None else "required",
                            body,
                        ),
                        Paragraph(rich(p.meaning) or "—", body),
                    ]
                )
            t = Table(rows, colWidths=[30 * mm, 32 * mm, 22 * mm, 82 * mm], repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8.4),
                        ("FONTSIZE", (0, 1), (-1, -1), 8.4),
                        ("TEXTCOLOR", (0, 0), (-1, 0), navy),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, navy),
                        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.Color(0.85, 0.85, 0.88)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            block.append(t)
        else:
            block.append(Paragraph("None.", body))

        block.append(blank_line)
        block.append(Paragraph("Output parameters", label))
        out_bits = []
        if e.returns:
            out_bits.append(f'<font face="DejaVuSansMono" size="8.4">{esc(e.returns)}</font>')
        if e.returns_meaning:
            out_bits.append(rich(e.returns_meaning))
        block.append(Paragraph(" — ".join(out_bits) if out_bits else "None.", body))

        if e.sources:
            block.append(blank_line)
            block.append(Paragraph("Sources", label))
            for s in e.sources:
                block.append(Paragraph("• " + rich(s), src_style))

        # shrink-to-fit so every function occupies exactly one page
        story.append(KeepInFrame(frame_w, frame_h - 14 * mm, block, mode="shrink", hAlign="LEFT"))
        story.append(PageBreak())

    doc.multiBuild(story)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} functions")


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    build_pdf(collect_entries())


if __name__ == "__main__":
    main()
