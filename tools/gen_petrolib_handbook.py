#!/usr/bin/env python3
"""Generate docs/Petrophysics_Handbook.pdf — the petrolib user manual.

Title page, an alphabetical table of contents of every public petrolib
function, and one section per function giving its purpose, input parameters,
output, and the full SPWLA *Petrophysics* journal citations resolved from the
module References sections.  Requires reportlab:

    pip install reportlab
    python tools/gen_petrolib_handbook.py
"""

from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Petrophysics_Handbook.pdf"

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


@dataclass
class Entry:
    name: str
    module: str
    kind: str  # "function" | "class"
    signature: str
    purpose: str
    params: list[Param] = field(default_factory=list)
    returns: str = ""
    sources: list[str] = field(default_factory=list)
    methods: list[tuple[str, str]] = field(default_factory=list)


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
    """Return (purpose prose, src tags) from a function docstring."""
    # rejoin tags wrapped across lines ("src2024_10/\n    ml_permeability")
    joined = re.sub(r"(src20\d\d_\d\d/)\s*\n\s*", r"\1", doc)
    tags: list[str] = []
    for tok in re.findall(r"src20\d\d_\d\d(?:/[A-Za-z0-9_]+)?", joined):
        if tok not in tags:
            tags.append(tok)
    purpose = re.sub(r"Sources?:.*?(?=\n\n|\Z)", "", joined, flags=re.S)
    paragraphs = [" ".join(seg.split()) for seg in purpose.split("\n\n") if seg.strip()]
    return "\n\n".join(paragraphs), tags


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


def signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef, params: list[Param]) -> str:
    bits: list[str] = []
    star_done = False
    for p in params:
        if p.kw_only and not star_done and not p.name.startswith("*"):
            bits.append("*")
            star_done = True
        if p.name.startswith("*"):
            star_done = True
        bits.append(p.name + (f"={p.default}" if p.default is not None else ""))
    return f"{node.name}({', '.join(bits)})"


def collect_entries() -> tuple[list[Entry], dict[str, dict[str, str]]]:
    entries: list[Entry] = []
    all_refs: dict[str, dict[str, str]] = {}
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
        all_refs[modname] = refs
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                doc = ast.get_docstring(node) or ""
                purpose, tags = split_purpose_sources(doc)
                params = params_of(node)
                ret = ast.unparse(node.returns) if node.returns else ""
                entries.append(
                    Entry(
                        name=node.name,
                        module=modname,
                        kind="function",
                        signature=signature_of(node, params),
                        purpose=purpose,
                        params=params,
                        returns=pretty_annotation(ret),
                        sources=[refs.get(t, t) for t in tags],
                    )
                )
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                purpose, tags = split_purpose_sources(doc)
                init = next(
                    (
                        n
                        for n in node.body
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    ),
                    None,
                )
                params = params_of(init) if init else []
                methods = []
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if m.name.startswith("_"):
                            continue
                        mdoc = (ast.get_docstring(m) or "").split("\n\n")[0]
                        methods.append((signature_of(m, params_of(m)), " ".join(mdoc.split())))
                sig = f"{node.name}({', '.join(p.name for p in params)})"
                entries.append(
                    Entry(
                        name=node.name,
                        module=modname,
                        kind="class",
                        signature=sig,
                        purpose=purpose,
                        params=params,
                        returns=f"{node.name} instance",
                        sources=[refs.get(t, t) for t in tags],
                        methods=methods,
                    )
                )
    entries.sort(key=lambda e: (e.name.lower(), e.module))
    return entries, all_refs


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

    body = ParagraphStyle("Body", fontName="DejaVuSans", fontSize=9.2, leading=12.4, spaceAfter=4)
    label = ParagraphStyle(
        "Label",
        parent=body,
        fontName="DejaVuSans-Bold",
        fontSize=9.2,
        spaceBefore=5,
        spaceAfter=2,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=body,
        fontName="DejaVuSansMono",
        fontSize=8.4,
        leading=10.8,
        backColor=colors.Color(0.955, 0.955, 0.965),
        borderPadding=3,
        spaceAfter=5,
    )
    func_heading = ParagraphStyle(
        "FuncHeading",
        fontName="DejaVuSans-Bold",
        fontSize=12.5,
        leading=15,
        spaceBefore=14,
        spaceAfter=1,
        textColor=colors.Color(0.10, 0.20, 0.36),
    )
    mod_line = ParagraphStyle(
        "ModLine",
        parent=body,
        fontName="DejaVuSans-Oblique",
        fontSize=8.4,
        textColor=colors.Color(0.35, 0.35, 0.40),
        spaceAfter=4,
    )
    src_style = ParagraphStyle(
        "Src", parent=body, fontSize=8.6, leading=11.4, leftIndent=10, spaceAfter=2
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
    frame = Frame(22 * mm, 20 * mm, A4[0] - 44 * mm, A4[1] - 40 * mm, id="main")

    def on_page(canvas: object, _doc: object) -> None:
        canvas.saveState()
        canvas.setFont("DejaVuSans", 8)
        canvas.setFillColor(colors.Color(0.35, 0.35, 0.40))
        canvas.drawString(22 * mm, A4[1] - 13 * mm, "Petrophysics Handbook")
        canvas.drawRightString(A4[0] - 22 * mm, A4[1] - 13 * mm, "petrolib reference")
        canvas.drawCentredString(A4[0] / 2, 11 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.addPageTemplates(
        [
            PageTemplate(id="title", frames=[frame]),
            PageTemplate(id="normal", frames=[frame], onPage=on_page),
        ]
    )

    story: list[object] = []

    # ---- title page
    story.append(Spacer(1, 70 * mm))
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
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "User manual for <b>petrolib</b> — the common petrophysics library of the "
            "Petrophysics_Code repository",
            ParagraphStyle("Sub", fontName="DejaVuSans", fontSize=13, leading=18, alignment=1),
        )
    )
    story.append(Spacer(1, 55 * mm))
    n_funcs = sum(1 for e in entries if e.kind == "function")
    n_classes = len(entries) - n_funcs
    story.append(
        Paragraph(
            f"{n_funcs} functions and {n_classes} classes across "
            f"{len({e.module for e in entries})} modules.<br/>"
            "Generated from the petrolib docstrings by "
            "tools/gen_petrolib_handbook.py.<br/>"
            "Source articles are published in <i>Petrophysics</i>, the journal of the "
            "Society of Petrophysicists and Well Log Analysts (SPWLA).",
            ParagraphStyle(
                "Col",
                fontName="DejaVuSans",
                fontSize=9,
                leading=13,
                alignment=1,
                textColor=colors.Color(0.35, 0.35, 0.40),
            ),
        )
    )
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

    # ---- one section per entry
    for i, e in enumerate(entries):
        key = f"e{i}"
        suffix = " (class)" if e.kind == "class" else ""
        head = Paragraph(f'<a name="{key}"/>{esc(e.name)}{suffix}', func_heading)
        head._toc_key = key
        head._toc_text = f"{e.name}{suffix}  ({e.module})"
        block: list[object] = [head, Paragraph(esc(e.module), mod_line)]
        block.append(Paragraph(esc(e.signature), mono))
        if e.purpose:
            block.append(Paragraph("Purpose", label))
            for para in e.purpose.split("\n\n"):
                block.append(Paragraph(rich(para), body))
        if e.params:
            block.append(Paragraph("Input parameters", label))
            rows = [["Parameter", "Type", "Default"]]
            for p in e.params:
                rows.append(
                    [
                        Paragraph(
                            f'<font face="DejaVuSansMono" size="8.4">{esc(p.name)}</font>'
                            + (" (kw-only)" if p.kw_only else ""),
                            body,
                        ),
                        Paragraph(esc(p.annotation) or "—", body),
                        Paragraph(
                            esc(p.default) if p.default is not None else "required",
                            body,
                        ),
                    ]
                )
            t = Table(rows, colWidths=[52 * mm, 72 * mm, 42 * mm], repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8.6),
                        ("FONTSIZE", (0, 1), (-1, -1), 8.6),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.Color(0.10, 0.20, 0.36)),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.Color(0.10, 0.20, 0.36)),
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
            block.append(Paragraph("Input parameters", label))
            block.append(Paragraph("None.", body))
        block.append(Paragraph("Output", label))
        block.append(Paragraph(esc(e.returns) or "None (procedure).", body))
        if e.methods:
            block.append(Paragraph("Methods", label))
            for msig, msum in e.methods:
                block.append(
                    Paragraph(
                        f'<font face="DejaVuSansMono" size="8.4">{esc(msig)}</font>'
                        + (f" — {rich(msum)}" if msum else ""),
                        body,
                    )
                )
        if e.sources:
            block.append(Paragraph("Sources", label))
            for s in e.sources:
                block.append(Paragraph("• " + rich(s), src_style))
        story.extend(block)

    doc.multiBuild(story)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} entries")


def main() -> None:
    entries, _ = collect_entries()
    build_pdf(entries)


if __name__ == "__main__":
    main()
