"""Command-line interface: one-shot mode and an interactive REPL.

Both modes stream the run: assistant text is printed as it arrives, and tool
calls / results are shown as single dim lines so the loop is observable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agents.exceptions import AgentsException
from openai import OpenAIError
from openai.types.responses import ResponseTextDeltaEvent, ResponseInputItemParam
from dotenv import load_dotenv

from . import tools
from .agent import build_agent, run_agent

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="general-research",
        description="A simple general-purpose quant research and coding agent (OpenAI Agents SDK + DeepSeek).",
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
        "-y",
        "--yes",
        action="store_true",
        help="run shell commands without asking for approval",
    )
    return parser.parse_args()


def _short(text: str, limit: int = 200) -> str:
    # text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


async def _run_with_output(agent, items: list) -> list[ResponseInputItemParam]:
    _DIM = "\x1b[2m"
    _RESET = "\x1b[0m"

    result = run_agent(agent, items)
    text = False

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                text = True
        else:
            if text:
                print("\n", flush=True)
                text = False

            if event.type == "run_item_stream_event":
                item = event.item
                if item.type == "message_output_item":
                    raw = item.raw_item
                    for content in raw.content:
                        if content.type == "output_text":
                            pass
                        elif content.type == "refusal":
                            print(content.refusal, flush=True)

                elif item.type == "reasoning_item":
                    raw = item.raw_item
                    for summary in raw.summary:
                        print(f"{_DIM}[thinking] {summary.text}{_RESET}", flush=True)

                elif item.type == "tool_call_item":
                    raw = item.raw_item
                    name = getattr(raw, "name", "?")
                    args = getattr(raw, "arguments", "")
                    print(f"\n{_DIM}[tool] {name} {_short(args)}{_RESET}", flush=True)

                elif item.type == "tool_call_output_item":
                    print(f"{_DIM}[tool] ->", flush=True)
                    print(f"{_DIM}{_short(item.output)}{_RESET}", flush=True)

                else:
                    print(f"{_DIM}[other] {item.type}{_RESET}", flush=True)

            elif event.type == "agent_updated_stream_event":
                print(f"{_DIM}[agent] {event.new_agent.name}{_RESET}", flush=True)

    print()
    return result.to_input_list()


async def _oneshot(agent, prompt: str) -> None:
    await _run_with_output(agent, [{"role": "user", "content": prompt}])


async def _repl(agent) -> None:
    print("general-research — type a task, '/exit' to quit, '/clear' to reset history.")
    items: list = []

    while True:
        try:
            user = input("\n> ").strip()
            print()
        except EOFError:
            break

        if not user:
            continue

        if user == "/exit":
            break

        if user == "/clear":
            items = []
            print("(history cleared)")
            print()
            continue

        items.append({"role": "user", "content": user})
        try:
            items = await _run_with_output(agent, items)
        except (AgentsException, OpenAIError) as e:
            # Drop the failed turn from history and keep the REPL alive.
            items.pop()
            print(f"\nerror: {e}", file=sys.stderr)


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
            asyncio.run(_oneshot(agent, args.prompt))
        else:
            asyncio.run(_repl(agent))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
