"""Agent definition: instructions, model wiring, and tool set.

DeepSeek implements the OpenAI Responses API (for `deepseek-v4-flash`), so the
Agents SDK talks to it through `OpenAIResponsesModel` — the SDK's primary
surface — rather than the older Chat Completions adapter. This buys native
reasoning items (thinking text streams as `response.reasoning_text.delta`
instead of being smuggled through a non-standard `reasoning_content` field)
and the server-side `web_search` tool, which the Chat Completions layer could
not offer.

DeepSeek's implementation is stateless: `previous_response_id`, `conversation`
and `store` are unsupported, so the full item list is resent every turn — which
is exactly what the SDK does by default here. Reasoning items are echoed back
as plain text (`encrypted_content` is never populated), and the server rebuilds
a `web_search_call`'s results from its id when the item is passed back verbatim.
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from agents import ModelSettings, RunResultStreaming, Runner, Agent, OpenAIResponsesModel, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types import Reasoning

from .tools import ALL_TOOLS, _shell_executable

_INSTRUCTIONS = """\
You are a quant researcher and coding agent operating on the user's machine. You need to help the user with investment advice, and (crucially) perform experiments with Rust and Python code when necessary.

# Environment

- Operating system: {os_info}
- Shell for the `run_command` tool: {shell}
- Working directory at the start of this conversation: {cwd}
- Date at start of this conversation: {today}

# General tips

- If any link appears to lead to an important or information-dense piece of documentation (e.g. overview, tutorial, interface spec), follow it. More knowledge helps you take informed actions.
- `web_search` runs on the server: a single call may issue several queries and open pages on its own. Use `web_fetch` when you already know the exact URL you want, or need a page's raw text (e.g. a CSV or JSON endpoint) rather than a summary.
- Make sure to read and understand any code you are trying to run: know its capabilities, limitations and estimate its performance. Make sure inputs are correct, and the logic is sound.
- Keep consistent style in your code.
- If a command can take more than a few seconds to run, you can use `run_command` with `wait_seconds = 0` to run it in the background, then use `check_command` to check for its outputs periodically. This also allows for running programs in parallel.
- When something is running, you can `wait` for shorter periods of time (e.g., 5-30 seconds) to see if it is making progress at an expected pace. After that, you can repeatedly `wait` for longer periods of time.
- When done, summarize what you changed and how it was verified. If something failed or was skipped, say so plainly.

# Financial tips

- Prefer objective, data-based analysis whenever possible. If there is a lack of data, try `web_search` and `web_fetch` to find a data source, then download and preprocess it.
- Given a new data source, you can do descriptive statistics (and visualizations for the user) using custom Python scripts, if needed.
- Be mindful of market complexity and risk: financial markets can be unintuitive, so never be too confident about a judgment or explanation. If a claim can be verified against data, do so.
- Do be aware and honest that even large amounts of data may not fully support a claim, even with classical statistical significance based on i.i.d. and normality assumptions (financial data is often fat-tailed, and panels have cross-sectional correlations).
- You can learn and implement advanced statistical tests (or use existing packages) if needed.

# Notes on TradingFlow (applies if the current working directory is in the `tradingflow` project)

Whenever there is a need to experiment with a complex trading strategy (where hand-written scripts are likely slow to run, or otherwise suboptimal), you can choose to use TradingFlow. In which case, the following tips may help:

- The TradingFlow project is a quantitative trading backtesting framework written in Rust.
- Its code should be self-documenting: read module-level docstrings, starting from crate roots and follow the doc links to get an idea of how everything works first.
- TradingFlow should come with some analysis and strategy examples, which you can use as templates for your own experiments. If explicitly allowed, you can also modify the examples to suit your needs.
- Ensure you are running experiments with `release` profile (`cargo run --release`) so that compiler optimization is enabled. If a single run seems to be taking more than 30 minutes to complete, consider reducing the data range or optimizing the implementation. If this is impossible or inconvenient, proceed by waiting anyway.

If you decide to use TradingFlow, always read its `README.md` and the crate roots `crates/tradingflow/src/lib.rs`, `crates/tradingflow-data/src/lib.rs`, `crates/tradingflow-graph/src/lib.rs` and understand its overall design first, before going back to the user's request.
"""


def build_agent(base_url: str, api_key: str, model: str) -> Agent:
    """Construct the coding agent against a DeepSeek (or compatible) endpoint."""

    # The SDK's tracing exporter uploads to the OpenAI platform, which we are
    # not using; disable it so it does not warn about a missing OpenAI key.
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    agent = Agent(
        name="general-research",
        instructions=_INSTRUCTIONS.format(
            os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
            shell=_shell_executable(),
            cwd=Path.cwd(),
            today=date.today().isoformat(),
        ),
        model=OpenAIResponsesModel(model=model, openai_client=client),
        # No `summary=`: DeepSeek accepts the field but never generates summaries,
        # emitting the raw thinking text as the reasoning item's content instead.
        model_settings=ModelSettings(reasoning=Reasoning(effort="high")),
        tools=ALL_TOOLS,
    )
    return agent


def run_agent(agent: Agent, items: list) -> RunResultStreaming:
    """Execute one run to completion, return the new history."""

    result = Runner.run_streamed(agent, input=items, max_turns=None)
    return result
