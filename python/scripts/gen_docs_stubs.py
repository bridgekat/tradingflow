"""Generate mkdocstrings stub pages for the tradingflow Python package.

Walks `python/src/tradingflow/` and writes one `:::<module>` stub per
module into `python/docs/`, mirroring the directory layout. Runs
before `zensical build` to keep the docs tree in sync with the source
tree automatically — no manual maintenance of per-module stubs.
"""

from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tradingflow"
DOCS = ROOT / "docs"


def _iter_stubs() -> Iterator[tuple[str, Path]]:
    """Yield `(module_fqn, output_path)` pairs for every public module."""
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
        else:
            stem = path.stem
            fqn = ".".join(["tradingflow", *dir_parts, stem])
            out = DOCS.joinpath(*dir_parts, f"{stem}.md")
        yield fqn, out


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

    for fqn, out in _iter_stubs():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f":::{fqn}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
