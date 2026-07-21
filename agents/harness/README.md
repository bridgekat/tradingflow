# `general-research`

A simple general-purpose quant research and coding agent built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), pointed at
the DeepSeek API. Self-contained; independent of the rest of the repository.

## Why the Agents SDK (and not the Responses API directly)

DeepSeek's endpoint is OpenAI-compatible at the **Chat Completions** layer only
— it does not implement the Responses API. The Agents SDK is the newer,
supported surface: agent-level code (tools, runner, streaming, sessions) is
written once against the SDK, and the provider transport is swapped via
`OpenAIChatCompletionsModel`, the SDK's documented adapter for third-party
OpenAI-compatible providers. If DeepSeek ships a Responses endpoint later,
only `general_research/agent.py` changes.

## Setup

```console
$ cd agents/harness
$ uv sync                      # or: python -m venv .venv && pip install -e .
$ copy .env.example .env       # then put your real DEEPSEEK_API_KEY in .env
```

## Usage

Run from the directory you want the agent to work in (tools resolve relative
paths against the current working directory):

```console
$ general-research                                       # interactive REPL
$ general-research -p "add a --verbose flag to cli.py"   # one-shot task
$ general-research --model deepseek-v4-pro               # thinking model
$ general-research -y                                    # don't ask before running shell commands
```

In the REPL: `exit`/`quit` to leave, `/clear` to reset conversation history.

Configuration comes from flags or environment (a `.env` in the working
directory is loaded automatically):

| Variable            | Default                    |                            |
| ------------------- | -------------------------- | -------------------------- |
| `DEEPSEEK_API_KEY`  | — (required)               | API key                    |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `DEEPSEEK_MODEL`    | `deepseek-v4-flash`        | `deepseek-v4-pro` = thinking |

(The legacy `deepseek-chat` / `deepseek-reasoner` names retire 2026-07-24.)

## Tool set

| Tool            | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `wait`          | Wait for a specified duration or keypress                  |
| `run_command`   | Shell command (asks for approval unless `-y`); commands outliving `wait_seconds` continue as background jobs |
| `check_command` | Status + incremental output of a background job            |
| `kill_command`  | Kill a background job (and its process tree)               |
| `list_dir`      | List directory entries                                     |
| `read_file`     | Read a file with line numbers (paged via `offset`/`limit`) |
| `write_file`    | Create/overwrite a file                                    |
| `edit_file`     | Exact-string replacement (unique match required)           |
| `glob`          | Find files by glob (`rg --files`, .gitignore-aware)        |
| `grep`          | Regex search via ripgrep (.gitignore-aware, context lines) |
| `web_fetch`     | Fetch a page as plain text (paged via `offset`)            |
| `web_search`    | Web search (keyless, DuckDuckGo via `ddgs`)                |

All tools return errors as strings so the model can observe and recover;
outputs are truncated at fixed caps to protect the context window.

### Why not the SDK's built-in tools?

The Agents SDK's built-in `WebSearchTool` / `FileSearchTool` /
`CodeInterpreterTool` are *hosted* tools: they execute server-side on the
OpenAI platform and require the Responses API backend, so they do not work
against DeepSeek (or any Chat Completions provider). DeepSeek does offer
server-side web search, but only on its Anthropic-compatible endpoint
(`/anthropic`, via `server_tool_use`), which the OpenAI SDK does not speak.
Hence `web_search` / `web_fetch` are plain function tools that run locally.
If you point this harness at OpenAI itself one day, you can swap them for
`WebSearchTool()`.

## Caveats

- There is no sandbox: the agent can touch anything your user account can.
  Shell commands require interactive approval unless you pass `-y`.
- SDK tracing is disabled (it would try to upload traces to the OpenAI
  platform, which this harness does not use).
