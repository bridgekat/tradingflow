**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. It is currently implemented in Python with NumPy.

# Features

- **Composable modules:** almost everything is a time series or a time series operator.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. When using AI coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

# Core Concepts

## Time Series

A [time series](src/tradingflow/series.py) is a sequence of uniformly typed elements indexed by strictly increasing `np.datetime64` timestamps, supporting integer indexing, slicing, timestamp lookups, and amortized O(1) appends.

An element in a time series can be a scalar, vector, matrix or higher-dimensional array with a fixed [`numpy.dtype`](https://numpy.org/doc/stable/reference/arrays.dtypes.html) and shape. Elements in a time series are internally stored in a single [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). This design is simple but not as flexible as Pandas-style data frames, which can contain columns of different types. To support such data, multiple time series must be created for different types.

## Sources

A [source](src/tradingflow/source.py) is used to generate values into a time series via asynchronous inputs from external data sources. A source must implement its `subscribe()` method, which returns two asynchronous iterators: one for historical `(timestamp, value)` tuples and one for live `value` updates. They should generate two complementary, non-overlapping segments of the same time series, split at some instant during the execution of `subscribe()`.

Sources are typically raw market data or pre-computed factors. Examples include:

- **Order flows** (e.g. transactions containing instrument name, price, quantity, etc.)
- **Snapshot prices** (e.g. arrays of all instrument prices updated at the end of each trading day)
- **Financial reports** (e.g. arrays of all balance sheet fields updated when a new financial report is released)

## Operators

An [operator](src/tradingflow/operator.py) is used to generate values into a time series via computations on other time series. An operator must implement its `compute()` method, which takes the current timestamp, a tuple of input time series and an optional mutable hidden state, and returns the updated output value. The `compute()` method is called to generate a new element from input time series when any of them is updated.

Operators are the reusable building blocks of trading strategies. Examples include:

- **Formulaic factors** (e.g. 20-day moving average of instrument prices)
- **Model predictions** (e.g. from a regression model predicting future instrument returns, periodically retrained on historical data)
- **Target positions** (e.g. periodically recomputed by mean-variance portfolio optimization on some forecasted returns and covariances)
- **Actual positions** (e.g. calculated from order execution history)
- **Trading signals** (e.g. differences between target and actual positions)
- **Performance metrics** (e.g. cumulative returns calculated from past positions)

## Scenarios

A [scenario](src/tradingflow/scenario.py) is a collection of time series, each associated with either a source or an operator along with its input time series. Time series dependencies must be acyclic. It provides a `run()` method which consumes all source streams, coalesces source events (which are only required to have non-decreasing timestamps) so that update timestamps are strictly increasing, and for each event updates all affected downstream time series.

## Storage Policies (TODO)

Based on the scenario demand, a time series may admit one of the following storage policies:

- **Last:** only the most recent value is stored. Only available if historical values are never accessed.
- **Window:** all values are stored but oldest entries are eventually removed. Only available if all accesses are within a fixed time window.
- **Full:** all values are stored.

All time series are immutable: a value cannot be modified once it is generated.
