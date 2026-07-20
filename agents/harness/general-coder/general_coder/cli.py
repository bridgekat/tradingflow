"""Command-line interface: one-shot mode and an interactive REPL.

Both modes stream the run: assistant text is printed as it arrives, and tool
calls / results are shown as single dim lines so the loop is observable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import openai
from agents import Runner
from agents.exceptions import AgentsException
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from . import tools
from .agent import DEFAULT_BASE_URL, DEFAULT_MODEL, build_agent

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="general-coder",
        description="A simple general-purpose coding agent (OpenAI Agents SDK + DeepSeek).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="run a single task non-interactively instead of starting the REPL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="model name (default: %(default)s; use deepseek-v4-pro for the thinking model)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="maximum model turns per task (default: %(default)s)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="run shell commands without asking for approval",
    )
    return parser.parse_args()


def _oneline(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


async def _stream_turn(agent, items: list, max_turns: int) -> list:
    """Run one task to completion, printing events; returns the new history."""
    result = Runner.run_streamed(agent, input=items, max_turns=max_turns)
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                raw = item.raw_item
                name = getattr(raw, "name", "?")
                args = getattr(raw, "arguments", "")
                print(f"\n{_DIM}[tool] {name} {_oneline(args)}{_RESET}", flush=True)
            elif item.type == "tool_call_output_item":
                print(
                    f"{_DIM}[tool] -> {_oneline(str(item.output))}{_RESET}", flush=True
                )
    print()
    return result.to_input_list()


async def _repl(agent, max_turns: int) -> None:
    print("general-coder — type a task, 'exit' to quit, '/clear' to reset history.")
    items: list = []
    while True:
        try:
            user = input("\n> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        if user == "/clear":
            items = []
            print("(history cleared)")
            continue
        items.append({"role": "user", "content": user})
        try:
            items = await _stream_turn(agent, items, max_turns)
        except (AgentsException, openai.OpenAIError) as e:
            # Drop the failed turn from history and keep the REPL alive.
            items.pop()
            print(f"\nerror: {e}", file=sys.stderr)


async def _oneshot(agent, prompt: str, max_turns: int) -> None:
    await _stream_turn(agent, [{"role": "user", "content": prompt}], max_turns)


def main() -> None:
    load_dotenv()  # before _parse_args: argument defaults read the environment
    args = _parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("error: DEEPSEEK_API_KEY is not set (see .env.example).")
    tools.AUTO_APPROVE = args.yes
    agent = build_agent(api_key=api_key, model=args.model, base_url=args.base_url)
    try:
        if args.prompt:
            asyncio.run(_oneshot(agent, args.prompt, args.max_turns))
        else:
            asyncio.run(_repl(agent, args.max_turns))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
