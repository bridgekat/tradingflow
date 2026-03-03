# Instructions for AI contributors

This file is kept short and concise. It provides initial context for all agentic tasks.

## Overview

- This project, named "TradingFlow", is a lightweight library for quantitative investment research.

## File Structure

- In the [Python source directory](src/), every folder is a module, which must contain an `__init__.py` file as its entry point. Do not put too much code in module entry points; implementations should go into separate files.

## Documentation

- Write and maintain module-level docstrings in all module entry points. A module-level docstring should be a comprehensive summary of the module's functionalities, structure and public API, stating important invariants when there is any, so that another person/agent with little prior knowledge can skip reading implementation files most of the time.
- Write and maintain class-level docstrings for public classes when appropriate.
- Write and maintain function-level docstrings for public functions when appropriate.
- Keep docstrings concise and accurate to current code. When there is an inconsistency, update either the docstring or the code to resolve it.

## Exploration

- Read module entry points first, starting from [the root](src/__init__.py).
- Read implementation files only for details that are not obvious from module entry points, or when there appears to be an inconsistency.
- For the given task, existing code with similar functionalities may be read for reference before implementation.

## Code Style

- Add Python 3.12+ type annotations except for overly complex types or overloads.

## Scope

- This project is in early stage, so breaking changes are allowed. Keep logical consistency throughout the codebase when introducing a breaking change.
