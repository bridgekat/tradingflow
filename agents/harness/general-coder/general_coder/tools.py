"""File-system, search, shell, and web tools exposed to the agent.

Every tool returns a plain string — including error messages — so the model
can observe a failure and react to it instead of the whole run aborting.
Outputs are truncated at fixed caps to keep them from flooding the model's
context window.

The web tools are ordinary function tools rather than the SDK's built-in
`WebSearchTool`: hosted tools execute on the OpenAI platform and require the
Responses API backend, which DeepSeek does not provide. (DeepSeek's own
server-side search exists only on its Anthropic-compatible endpoint.)
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from html.parser import HTMLParser
import httpx
from agents import Tool, function_tool

# Caps on tool output size.
_MAX_READ_LINES = 2000
_MAX_LINE_CHARS = 1000
_MAX_OUTPUT_CHARS = 50_000
_MAX_MATCHES = 200
_MAX_GLOB_RESULTS = 500
_MAX_WEB_RESULTS = 20
_MAX_FETCH_CHARS = 20_000
_MAX_FETCH_BYTES = 5_000_000
_MAX_WAIT_SECONDS = 600

# Some sites reject non-browser user agents outright.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

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


def _sleep_until_keypress(seconds: float) -> bool:
    """Sleep for up to `seconds`; return True if a keypress ended it early.

    Keypresses can only be observed when stdin is a terminal; otherwise this
    is a plain uninterruptible sleep. Any keystrokes already buffered before
    the wait starts are discarded so a stale key cannot cancel it instantly.
    """

    deadline = time.monotonic() + seconds

    if not sys.stdin.isatty():
        time.sleep(seconds)
        return False

    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if msvcrt.kbhit():
                msvcrt.getwch()
                return True
            time.sleep(min(0.05, remaining))

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # deliver single keystrokes without waiting for Enter
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            readable, _, _ = select.select([fd], [], [], remaining)
            if readable:
                os.read(fd, 1)
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


@functools.lru_cache(maxsize=1)
def _shell_executable() -> str:
    """The shell `run_command` executes through. On Windows this prefers
    PowerShell 7+ (`pwsh`), then Windows PowerShell, then cmd.exe; on POSIX it
    is /bin/sh. Cached so the model's instructions and `run_command` can never
    disagree. Reported to the model so it writes the right dialect."""
    if os.name == "nt":
        for name in ("pwsh", "powershell"):
            exe = shutil.which(name)
            if exe:
                return exe
        return os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
    return "/bin/sh"


def _shell_argv(command: str) -> list[str]:
    """Build the argv that runs `command` through `run_command_shell()`."""
    shell = _shell_executable()
    match Path(shell).stem.lower():
        case "pwsh" | "powershell":
            # -NoProfile: fast and reproducible startup; -NonInteractive: fail
            # instead of hanging when something prompts for input.
            return [shell, "-NoProfile", "-NonInteractive", "-Command", command]
        case "cmd":
            return [shell, "/d", "/s", "/c", command]
        case _:
            return [shell, "-c", command]


def _rg_executable() -> str | None:
    """Locate ripgrep: the system PATH first, then the binary the `ripgrep`
    pip package installs next to the Python interpreter."""
    exe = shutil.which("rg")
    if exe:
        return exe
    bundled = Path(sys.executable).parent / ("rg.exe" if os.name == "nt" else "rg")
    return str(bundled) if bundled.is_file() else None


@function_tool
def wait(seconds: float) -> str:
    """Wait for a given duration, e.g. for an external process or a rate limit.

    While waiting, the user can press any key to end the wait immediately.
    The result reports how long was actually waited and whether the wait ran
    to completion or was cut short by the user.

    Args:
        seconds: How long to wait, in seconds (at most 600; wait repeatedly for longer pauses).
    """

    if not seconds > 0:
        return "Error: seconds must be positive."
    if seconds > _MAX_WAIT_SECONDS:
        return f"Error: seconds must be at most {_MAX_WAIT_SECONDS}; wait repeatedly for longer pauses."

    if sys.stdin.isatty():
        print(f"\n  waiting {seconds:g}s - press any key to end the wait early", flush=True)

    start = time.monotonic()
    try:
        stopped_by_user = _sleep_until_keypress(seconds)
    except OSError as e:
        return f"Error: {e}"
    elapsed = time.monotonic() - start

    if stopped_by_user:
        return f"waited for {elapsed:.1f} seconds (user ended the wait)"
    return f"waited for {elapsed:.1f} seconds"


@function_tool
def run_command(command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command in the working directory and return its output.

    Use this to build, test, lint, or inspect things the other tools cannot.
    The command runs through the shell named in your environment information;
    the user may be asked to approve it first.

    Args:
        command: The shell command line to execute.
        timeout_seconds: Kill the command after this many seconds.
    """

    if not AUTO_APPROVE:
        if not sys.stdin.isatty():
            return (
                "Error: command not run — approval required but stdin is not a terminal (rerun the harness with --yes)."
            )

        try:
            reply = input(f"\n  approve command? [y/N] {command}\n  > ").strip().lower()
        except EOFError:
            return "Error: command not run — could not read approval from stdin (rerun the harness with --yes)."

        if reply not in ("y", "yes"):
            return "Error: command rejected by the user."

    try:
        proc = subprocess.run(
            _shell_argv(command),
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
    numbered = "\n".join(f"{offset + i:6d} | {line[:_MAX_LINE_CHARS]}" for i, line in enumerate(window))
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
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
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

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    try:
        p.write_text(new_text, encoding="utf-8", newline="\n")
    except OSError as e:
        return f"Error writing {p}: {e}"

    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}."


@function_tool
def glob(pattern: str, path: str = ".") -> str:
    """List files matching a glob pattern (gitignore-style semantics).

    Respects .gitignore and skips hidden files. A bare pattern like `*.py`
    matches at any directory depth; patterns containing `/` (e.g.
    `src/**/*.rs`) are anchored to the search root.

    Args:
        pattern: Glob pattern, e.g. `*.py` or `src/**/test_*.rs`.
        path: Directory to search in, absolute or relative to the working directory.
    """

    exe = _rg_executable()
    if exe is None:
        return "Error: ripgrep not found (pip install ripgrep, or put `rg` on PATH)."

    root = _resolve(path)
    if not root.is_dir():
        return f"Error: {root} is not a directory."

    cmd = [exe, "--files", "--glob", pattern, "--", str(root)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error: ripgrep failed to run: {e}"

    if not proc.stdout:
        if proc.returncode in (0, 1):  # rg --files exits 1 when nothing was listed
            return f"No files match {pattern!r} in {root}."
        return f"Error: ripgrep failed: {proc.stderr.strip() or f'exit code {proc.returncode}'}"

    matches = sorted(proc.stdout.splitlines())
    shown = matches[:_MAX_GLOB_RESULTS]
    out = "\n".join(shown)
    if len(matches) > len(shown):
        out += f"\n... [{len(matches) - len(shown)} more matches; narrow the pattern]"

    return _truncate(out)


@function_tool
def grep(pattern: str, path: str = ".", glob: str = "", context: int = 0, ignore_case: bool = False) -> str:
    """Search file contents with ripgrep, like grep -rn.

    Respects .gitignore and skips hidden and binary files. Output lines are
    `path:line: text` (context lines use `path-line- text`).

    Args:
        pattern: Regular expression (Rust regex syntax) to search for.
        path: Directory (or single file) to search, absolute or relative.
        glob: Optional file filter, e.g. `*.py` or `src/**/*.rs`.
        context: Lines of context to show around each match.
        ignore_case: Match case-insensitively.
    """

    exe = _rg_executable()
    if exe is None:
        return "Error: ripgrep not found (pip install ripgrep, or put `rg` on PATH)."

    root = _resolve(path)
    if not root.exists():
        return f"Error: {root} does not exist."

    cmd = [exe, "--line-number", "--no-heading", "--color", "never", "--max-columns", str(_MAX_LINE_CHARS)]
    if glob:
        cmd += ["--glob", glob]
    if context > 0:
        cmd += ["--context", str(context)]
    if ignore_case:
        cmd.append("--ignore-case")
    cmd += ["--regexp", pattern, "--", str(root)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error: ripgrep failed to run: {e}"

    if not proc.stdout:
        if proc.returncode == 1:  # rg: 0 = matches, 1 = no matches, 2 = error
            return f"No matches for {pattern!r} under {root}."
        return f"Error: ripgrep failed: {proc.stderr.strip() or f'exit code {proc.returncode}'}"

    lines = proc.stdout.splitlines()
    if len(lines) > _MAX_MATCHES:
        lines = lines[:_MAX_MATCHES] + [f"... [stopped at {_MAX_MATCHES} lines; narrow the search]"]

    return _truncate("\n".join(lines))


class _HtmlText(HTMLParser):
    """Extracts the visible text of an HTML document."""

    _SKIP_TAGS = {"script", "style", "noscript", "template"}
    _BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
        "header", "footer", "nav", "pre", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP_TAGS:
            self._skip += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def text(self) -> str:
        collapsed = re.sub(r"[ \t]+", " ", "".join(self._parts))
        return re.sub(r"\n\s*\n+", "\n\n", collapsed).strip()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _HtmlText()

    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup; fall back to a crude tag strip
        return re.sub(r"<[^>]+>", " ", html)

    return parser.text()


@function_tool
def web_fetch(url: str, offset: int = 0) -> str:
    """Fetch a web page and return its visible text (HTML tags stripped).

    Args:
        url: The http(s) URL to fetch.
        offset: Character offset to continue reading a long page from.
    """

    if not url.startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported."

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    if resp.status_code >= 400:
        return f"Error: HTTP {resp.status_code} for {url}."

    if len(resp.content) > _MAX_FETCH_BYTES:
        return f"Error: response too large ({len(resp.content)} bytes)."

    ctype = resp.headers.get("content-type", "")
    text = _html_to_text(resp.text) if "html" in ctype else resp.text
    offset = max(offset, 0)
    window = text[offset : offset + _MAX_FETCH_CHARS]
    if not window:
        return f"{url} has {len(text)} characters of text; nothing at offset {offset}."

    remaining = len(text) - (offset + len(window))
    if remaining > 0:
        window += f"\n... [{remaining} more characters; continue with" f" offset={offset + len(window)}]"

    return window


@function_tool
def web_search(query: str, max_results: int = 8) -> str:
    """Search the web and return result titles, URLs, and snippets.

    Use `web_fetch` afterwards to read the full text of a promising result.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    from ddgs import DDGS  # imported lazily: constructing it opens an HTTP client

    # ddgs rotates across search providers; a transient failure on one is
    # common, and retrying usually lands on a different provider.
    n = max(1, min(max_results, _MAX_WEB_RESULTS))
    results = []
    for attempt in (1, 2):
        try:
            results = list(DDGS(timeout=15).text(query, max_results=n))
            break
        except Exception as e:  # the backends raise assorted network errors
            if attempt == 2:
                return f"Error: web search failed: {e}"
    if not results:
        return f"No results for {query!r}."

    blocks = []
    for i, r in enumerate(results, start=1):
        url = r.get("href") or r.get("url") or ""
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        blocks.append(f"{i}. {title}\n   {url}\n   {body}")

    return _truncate("\n".join(blocks))


ALL_TOOLS: list[Tool] = [
    wait,
    run_command,
    list_dir,
    read_file,
    write_file,
    edit_file,
    glob,
    grep,
    web_fetch,
    web_search,
]
