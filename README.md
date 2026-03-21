[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. The core runtime is implemented in Rust; a Python wrapper is provided for ease of use with Python's data science ecosystem.

# Features

- **Composable modules:** almost every piece of data is a time series, and almost every computation is a time series operator.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. When using AI coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

# Core Concepts

## Data stores

A [data store](src/store.rs) represents a time series. It holds a contiguous buffer of uniformly typed elements and a buffer of non-decreasing timestamps (nanoseconds since the UNIX epoch) associated to each element. The element type and shape (can be scalar, vector, matrix, or higher-dimensional array) are fixed at creation time.

A configurable **window size** controls retention:

- `window = 1` — stores only the most recent element.
- `window = N` — stores a fixed sliding window of the most recent `N` elements.
- `window = 0` — stores the full time series.

Cheap **views** can be created from a data store without allocation:

- `ElementView` / `ElementViewMut` — a single element.
- `SeriesView` — the full retained history.

## Sources

A [source](src/source.rs) feeds elements into a data store via asynchronous channels. A source must implement its `subscribe()` method, which returns two channel receivers: one for historical `(timestamp, event)` tuples and one for real-time events. They should generate two complementary, non-overlapping segments of the same time series, split at some instant during the execution of `subscribe()`.

Sources are typically raw market data or pre-computed factors. Examples include:

- **Order flows** (e.g. transactions containing instrument name, price, quantity, etc.)
- **Snapshot prices** (e.g. arrays of all instrument prices updated at the end of each trading day)
- **Financial reports** (e.g. arrays of all balance sheet fields updated when a new financial report is released)

## Operators

An [operator](src/operator.rs) reads from one or more input data stores and writes into new elements in some output data store. An operator must implement its `compute()` method, which takes the current state, input data store references, and a mutable output view, and returns whether a value was produced. The `compute()` method is called to generate a new output when any input is updated.

Each operator declares its required **minimum input window sizes**. Each data store's retention window is then computed as the maximum window size required by operators that depend on it, ensuring that all necessary historical data is retained for correct computation.

Operators are the reusable building blocks of trading strategies. Examples include:

- **Technical indicators** (e.g. 20-day moving average of instrument prices)
- **Model predictions** (e.g. from a regression model predicting future instrument returns, periodically retrained on historical data)
- **Target positions** (e.g. periodically recomputed by mean-variance portfolio optimization on some forecasted returns and covariances)
- **Trading signals** (e.g. differences between target and actual positions)
- **Performance metrics** (e.g. cumulative returns calculated from past positions)

## Scenarios

A [scenario](src/scenario/mod.rs) is a directed acyclic graph, whose nodes own data stores and whose edges are the operator dependencies. Each node can be associated with either a source or an operator. Dependencies must be acyclic.

The scenario provides a `run()` method which consumes all source streams (historical and live) and puts them in a queue, coalesces events at the same timestamp, and for each batch propagates updates through the graph in topological order. It makes sure that update timestamps are strictly increasing each batch.
