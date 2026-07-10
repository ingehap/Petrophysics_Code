#!/usr/bin/env python3
"""Generate docs/petrolib-api.md from the petrolib docstrings.

Walks every petrolib module, and emits one section per module with its
docstring summary and a table of the public API (classes, functions, and
UPPER_CASE constants) with each object's one-line summary.  Re-run after
adding or changing petrolib functions:

    python tools/gen_petrolib_api.py
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "petrolib-api.md"

HEADER = """\
# petrolib API reference

One-line summaries of the public API, generated from the docstrings by
[`tools/gen_petrolib_api.py`](../tools/gen_petrolib_api.py) — regenerate after
any petrolib change.  Full parameter documentation lives in the docstrings
(``help(petrolib.<module>.<function>)``); complete journal citations for every
``srcYYYY_MM`` source tag are in each module docstring's *References* section.

"""


def first_sentence(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    text = " ".join(line.strip() for line in para.splitlines())
    m = re.match(r"(.+?\.)(\s|$)", text)
    text = m.group(1) if m else text
    return text.replace("|", "\\|")


def first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    return " ".join(line.strip() for line in para.splitlines())


def signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    names: list[str] = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        names.append("*")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    sig = ", ".join(n for n in names if n != "self")
    if len(sig) > 58:
        sig = sig[: sig[:55].rfind(",")] + ", ..."
    return sig


def module_rows(tree: ast.Module, src_lines: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            rows.append(
                (f"`{node.name}({signature_of(node)})`", first_sentence(ast.get_docstring(node)))
            )
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            methods = ", ".join(
                f"`{m.name}`"
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not m.name.startswith("_")
            )
            summary = first_sentence(ast.get_docstring(node))
            if methods:
                summary = (summary + " Methods: " + methods + ".").strip()
            rows.append((f"`class {node.name}`", summary))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id.isupper():
                # gather the "#:" doc comment lines directly above the assignment
                doc_lines: list[str] = []
                i = node.lineno - 2
                while i >= 0 and src_lines[i].strip().startswith("#:"):
                    doc_lines.insert(0, src_lines[i].strip().removeprefix("#:").strip())
                    i -= 1
                rows.append((f"`{t.id}`", " ".join(doc_lines).replace("|", "\\|") or "constant"))
    return rows


def main() -> None:
    parts: list[str] = [HEADER]
    files = sorted(
        p
        for p in ROOT.glob("petrolib/**/*.py")
        if "__pycache__" not in str(p) and p.name not in ("__init__.py", "_compat.py")
    )
    toc: list[str] = ["## Modules", ""]
    sections: list[str] = []
    for pfile in files:
        rel = pfile.relative_to(ROOT)
        modname = str(rel.with_suffix("")).removeprefix("petrolib/").replace("/", ".")
        src = pfile.read_text()
        tree = ast.parse(src)
        summary = first_paragraph(ast.get_docstring(tree))
        anchor = "petrolib" + modname.replace(".", "")
        toc.append(f"- [`petrolib.{modname}`](#{anchor}) — {summary}")
        rows = module_rows(tree, src.splitlines())
        sec = [f"## `petrolib.{modname}`", "", summary, ""]
        if rows:
            sec += ["| Name | Summary |", "| --- | --- |"]
            sec += [f"| {name} | {desc} |" for name, desc in rows]
        sec.append("")
        sections.append("\n".join(sec))
    parts.append("\n".join(toc) + "\n")
    parts.extend(sections)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(files)} modules)")


if __name__ == "__main__":
    main()
