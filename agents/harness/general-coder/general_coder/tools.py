"""File-system, search, and shell tools exposed to the agent.

Every tool returns a plain string — including error messages — so the model
can observe a failure and react to it instead of the whole run aborting.
Outputs are truncated at fixed caps to keep them from flooding the model's
context window.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from agents import Tool, function_tool

# Caps on tool output size.
_MAX_READ_LINES = 2000
_MAX_LINE_CHARS = 1000
_MAX_OUTPUT_CHARS = 50_000
_MAX_MATCHES = 200
_MAX_GLOB_RESULTS = 500
_MAX_SEARCH_FILE_BYTES = 2_000_000

# Directories that are never worth listing matches from.
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".eggs",
}

# When False, `run_command` asks for confirmation on stdin before executing
# (or refuses if stdin is not a terminal). The CLI sets this from `--yes`.
AUTO_APPROVE = False


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated; {len(text) - limit} more characters]"


def _skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


@function_tool
def read_file(path: str, offset: int = 1, limit: int = _MAX_READ_LINES) -> str:
    """Read a text file, returning its contents with 1-based line numbers.

    Args:
        path: File path, absolute or relative to the working directory.
        offset: Line number to start reading from (1-based).
        limit: Maximum number of lines to return.
    """
    p = _resolve(path)
    if not p.is_file():
        return f"Error: {p} is not a file."
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading {p}: {e}"
    offset = max(offset, 1)
    window = lines[offset - 1 : offset - 1 + max(limit, 1)]
    if not window:
        return f"{p} has {len(lines)} lines; nothing at offset {offset}."
    numbered = "\n".join(
        f"{offset + i:6d} | {line[:_MAX_LINE_CHARS]}" for i, line in enumerate(window)
    )
    remaining = len(lines) - (offset - 1 + len(window))
    if remaining > 0:
        numbered += f"\n... [{remaining} more lines; continue with offset={offset + len(window)}]"
    return _truncate(numbered)


@function_tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Parent directories are created as needed. Prefer `edit_file` for small
    changes to existing files.

    Args:
        path: File path, absolute or relative to the working directory.
        content: Full new contents of the file.
    """
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
    except OSError as e:
        return f"Error writing {p}: {e}"
    return f"Wrote {len(content)} characters to {p}."


@function_tool
def edit_file(
    path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Replace an exact string in a file.

    `old_string` must match the file contents exactly (including whitespace),
    and must be unique in the file unless `replace_all` is set. Read the file
    first to get the exact text.

    Args:
        path: File path, absolute or relative to the working directory.
        old_string: Exact text to find.
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of requiring uniqueness.
    """
    p = _resolve(path)
    if not p.is_file():
        return f"Error: {p} is not a file."
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading {p}: {e}"
    count = text.count(old_string)
    if count == 0:
        return "Error: old_string not found in the file. Read the file and match its contents exactly."
    if count > 1 and not replace_all:
        return f"Error: old_string occurs {count} times; add more surrounding context to make it unique, or set replace_all=true."
    new_text = (
        text.replace(old_string, new_string)
        if replace_all
        else text.replace(old_string, new_string, 1)
    )
    try:
        p.write_text(new_text, encoding="utf-8", newline="\n")
    except OSError as e:
        return f"Error writing {p}: {e}"
    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}."


@function_tool
def list_dir(path: str = ".") -> str:
    """List the entries of a directory (directories first, with file sizes).

    Args:
        path: Directory path, absolute or relative to the working directory.
    """
    p = _resolve(path)
    if not p.is_dir():
        return f"Error: {p} is not a directory."
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except OSError as e:
        return f"Error listing {p}: {e}"
    lines = [f"{p}:"]
    for e in entries:
        if e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            try:
                size = e.stat().st_size
            except OSError:
                size = 0
            lines.append(f"  {e.name}  ({size} bytes)")
    return _truncate("\n".join(lines))


@function_tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern (`**` matches across directories).

    Args:
        pattern: Glob pattern, e.g. `**/*.py` or `src/**/test_*.rs`.
        path: Directory to search in, absolute or relative to the working directory.
    """
    p = _resolve(path)
    if not p.is_dir():
        return f"Error: {p} is not a directory."
    try:
        matches = [
            m for m in p.glob(pattern) if m.is_file() and not _skip(m.relative_to(p))
        ]
    except (OSError, ValueError) as e:
        return f"Error globbing {pattern!r} in {p}: {e}"
    if not matches:
        return f"No files match {pattern!r} in {p}."
    matches.sort()
    shown = matches[:_MAX_GLOB_RESULTS]
    out = "\n".join(str(m) for m in shown)
    if len(matches) > len(shown):
        out += f"\n... [{len(matches) - len(shown)} more matches; narrow the pattern]"
    return _truncate(out)


@function_tool
def search_files(pattern: str, path: str = ".", glob: str = "*") -> str:
    """Search file contents with a regular expression, like grep -rn.

    Args:
        pattern: Python regular expression to search for.
        path: Directory (or single file) to search, absolute or relative.
        glob: Filename filter, e.g. `*.py` (matched against the basename).
    """
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex {pattern!r}: {e}"
    root = _resolve(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [
            f
            for f in root.rglob(glob)
            if f.is_file() and not _skip(f.relative_to(root))
        ]
    else:
        return f"Error: {root} does not exist."
    hits: list[str] = []
    for f in sorted(files):
        try:
            if f.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text[:1024]:  # binary file
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{f}:{lineno}: {line.strip()[:_MAX_LINE_CHARS]}")
                if len(hits) >= _MAX_MATCHES:
                    hits.append(
                        f"... [stopped at {_MAX_MATCHES} matches; narrow the search]"
                    )
                    return _truncate("\n".join(hits))
    if not hits:
        return f"No matches for {pattern!r} under {root}."
    return _truncate("\n".join(hits))


@function_tool
def run_command(command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command in the working directory and return its output.

    Use this to build, test, lint, or inspect things the other tools cannot.
    The command runs through the system shell; the user may be asked to
    approve it first.

    Args:
        command: The shell command line to execute.
        timeout_seconds: Kill the command after this many seconds.
    """
    if not AUTO_APPROVE:
        if not sys.stdin.isatty():
            return "Error: command not run — approval required but stdin is not a terminal (rerun the harness with --yes)."
        try:
            reply = input(f"\n  approve command? [y/N] {command}\n  > ").strip().lower()
        except EOFError:
            return "Error: command not run — could not read approval from stdin (rerun the harness with --yes)."
        if reply not in ("y", "yes"):
            return "Error: command rejected by the user."
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds} seconds."
    except OSError as e:
        return f"Error running command: {e}"
    parts = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        parts.append("--- stdout ---\n" + proc.stdout)
    if proc.stderr:
        parts.append("--- stderr ---\n" + proc.stderr)
    return _truncate("\n".join(parts))


ALL_TOOLS: list[Tool] = [
    read_file,
    write_file,
    edit_file,
    list_dir,
    glob_files,
    search_files,
    run_command,
]
