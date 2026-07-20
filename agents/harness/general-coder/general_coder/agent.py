"""Agent definition: instructions, model wiring, and tool set.

DeepSeek's endpoint is OpenAI-compatible at the Chat Completions layer only
(it does not implement the Responses API), so the Agents SDK is configured
with its `OpenAIChatCompletionsModel` adapter — the documented way to run the
SDK against third-party providers. Agent-level code (tools, runner, streaming)
is identical either way; if DeepSeek ever ships a Responses endpoint, only
this module changes.
"""

from __future__ import annotations

from pathlib import Path

from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

from .tools import ALL_TOOLS

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

_INSTRUCTIONS = """\
You are a general-purpose coding agent operating on the user's machine.
The working directory is: {cwd}

Workflow:
1. Explore before you act: use list_dir / glob_files / search_files / read_file
   to understand the relevant code. Never guess file contents — read a file
   before editing it.
2. Make focused changes with edit_file (preferred for existing files) or
   write_file (new files or full rewrites).
3. Verify your work with run_command (build, tests, linters) when possible.

Rules:
- Paths may be absolute or relative to the working directory.
- Keep changes minimal and consistent with the surrounding code style.
- When done, summarize what you changed and how it was verified. If something
  failed or was skipped, say so plainly.
"""


def build_agent(
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> Agent:
    """Construct the coding agent against a DeepSeek (or compatible) endpoint."""
    # The SDK's tracing exporter uploads to the OpenAI platform, which we are
    # not using; disable it so it does not warn about a missing OpenAI key.
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return Agent(
        name="general-coding-agent",
        instructions=_INSTRUCTIONS.format(cwd=Path.cwd()),
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
        tools=ALL_TOOLS,
    )
