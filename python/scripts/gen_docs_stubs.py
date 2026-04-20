"""Generate mkdocstrings stub pages for the tradingflow Python package.

Walks `python/src/tradingflow/` and writes one `:::<module>` stub per
module into `python/docs/`, mirroring the directory layout. Runs
before `zensical build` to keep the docs tree in sync with the source
tree automatically — no manual maintenance of per-module stubs.

Package `__init__.py` files in this project contain only re-exports
(no native class/function definitions), so their stubs disable member
rendering (`members: false`) to avoid duplicating full docs for each
re-exported symbol — the canonical docs live on the leaf-module pages,
and the package page shows only the module overview docstring with
cross-links.
"""

from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tradingflow"
DOCS = ROOT / "docs"


def _iter_stubs() -> Iterator[tuple[str, Path, bool]]:
    """Yield `(module_fqn, output_path, is_package)` for every public module."""
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        dir_parts = rel.parts[:-1]
        file_name = rel.parts[-1]
        if any(p.startswith("_") for p in dir_parts):
            continue
        if file_name != "__init__.py" and file_name.startswith("_"):
            continue
        if file_name == "__init__.py":
            fqn = ".".join(["tradingflow", *dir_parts])
            out = DOCS.joinpath(*dir_parts, "index.md")
            is_package = True
        else:
            stem = path.stem
            fqn = ".".join(["tradingflow", *dir_parts, stem])
            out = DOCS.joinpath(*dir_parts, f"{stem}.md")
            is_package = False
        yield fqn, out, is_package


def _render(fqn: str, is_package: bool) -> str:
    if is_package:
        return f":::{fqn}\n    options:\n      members: false\n"
    return f":::{fqn}\n"


def main() -> None:
    for md in DOCS.rglob("*.md"):
        md.unlink()
    for d in sorted(
        (p for p in DOCS.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass

    for fqn, out, is_package in _iter_stubs():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render(fqn, is_package), encoding="utf-8")


if __name__ == "__main__":
    main()
