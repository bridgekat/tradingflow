# Instructions for AI contributors

This file is kept short and concise. It provides initial context for all agentic tasks.

## Overview

- This project, named "TradingFlow", is a lightweight library for quantitative investment research.
- The core part is implemented in Rust for performance. A Python wrapper is provided for ease of use with Python's data science ecosystem.

## File Structure

- Rust sources live in [`src/`](src/) (the crate root). Every folder is a module with a `mod.rs` or `lib.rs` entry point.
- Python sources live in [`python/src/tradingflow/`](python/src/tradingflow/). Every folder is a module with an `__init__.py` entry point.
- The project is built with [Maturin](https://www.maturin.rs/): `pip install -e ".[dev]"` compiles the Rust crate and installs the Python package together.

## Documentation

- Write and maintain module-level docstrings in all module entry points. Such a docstring should be a comprehensive summary of the module's functionalities, structure and public API, stating important invariants when there is any, so that another person/agent with little prior knowledge can skip reading implementation files most of the time. They serve as an index of the codebase.
- Write and maintain class-level docstrings for public classes when appropriate.
- Write and maintain function-level docstrings for public functions when appropriate.
- Keep docstrings concise and accurate to current code. When there is an inconsistency, update either the docstring or the code to resolve it.

## Exploration

- Read module entry points first, starting from [the Rust root](src/lib.rs) or [the Python root](python/src/tradingflow/__init__.py).
- Read implementation files only for details that are not obvious from module entry points, or when there appears to be an inconsistency.
- For the given task, existing code with similar functionalities may be read for reference before implementation.

## Code Style

- Do not put too much code in module entry points; implementations should go into separate files.
- All Rust docstrings should follow [rustdoc](https://doc.rust-lang.org/rustdoc/) syntax and style.
- All Python docstrings should follow Markdown (not reStructuredText) for base syntax, [mkdocstrings](https://mkdocstrings.github.io/usage/) for cross-references syntax and [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) only for sections style. This is supported by [Zensical](https://zensical.org/docs/), the new documentation generator we use.
- Add Python 3.12+ type annotations except for overly complex types or overloads.

## Examples

- Examples live in [`python/examples/`](python/examples/) and use data from external sources.
- Install optional dependencies for examples: `pip install -e ".[examples]"`.

## Scope

- This project is in early stage, so breaking changes are allowed. Keep logical consistency throughout the codebase when introducing a breaking change.
