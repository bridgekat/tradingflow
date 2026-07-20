# `general-coder`

A simple general-purpose coding agent built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), pointed at
the DeepSeek API. Self-contained; independent of the rest of the repository.

## Why the Agents SDK (and not the Responses API directly)

DeepSeek's endpoint is OpenAI-compatible at the **Chat Completions** layer only
— it does not implement the Responses API. The Agents SDK is the newer,
supported surface: agent-level code (tools, runner, streaming, sessions) is
written once against the SDK, and the provider transport is swapped via
`OpenAIChatCompletionsModel`, the SDK's documented adapter for third-party
OpenAI-compatible providers. If DeepSeek ships a Responses endpoint later,
only `general_coder/agent.py` changes.

## Setup

```console
$ cd agents/harness/general-coder
$ python -m venv .venv
$ .venv\Scripts\activate       # Windows; `source .venv/bin/activate` elsewhere
$ pip install -e .
$ copy .env.example .env       # then put your real DEEPSEEK_API_KEY in .env
```

## Usage

Run from the directory you want the agent to work in (tools resolve relative
paths against the current working directory):

```console
$ general-coder                          # interactive REPL
$ general-coder -p "add a --verbose flag to cli.py"   # one-shot task
$ general-coder --model deepseek-v4-pro  # thinking model
$ general-coder -y                       # don't ask before running shell commands
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

| Tool           | Purpose                                                    |
| -------------- | ---------------------------------------------------------- |
| `read_file`    | Read a file with line numbers (paged via `offset`/`limit`) |
| `write_file`   | Create/overwrite a file                                    |
| `edit_file`    | Exact-string replacement (unique match required)           |
| `list_dir`     | List directory entries                                     |
| `glob_files`   | Find files by glob pattern                                 |
| `search_files` | Regex search across files (grep-like)                      |
| `run_command`  | Shell command (asks for approval unless `-y`)              |

All tools return errors as strings so the model can observe and recover;
outputs are truncated at fixed caps to protect the context window.

## Caveats

- There is no sandbox: the agent can touch anything your user account can.
  Shell commands require interactive approval unless you pass `-y`.
- SDK tracing is disabled (it would try to upload traces to the OpenAI
  platform, which this harness does not use).
