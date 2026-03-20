[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. It is currently implemented in Python with NumPy.

# Features

- **Composable modules:** almost every piece of data is an observable value or time series, and almost every computation is an operator on them.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. When using AI coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

# Core Concepts

## Observable values

An [observable value](python/tradingflow/observable.py) is a piece of data that can be updated at strictly increasing `np.datetime64` timestamps.

The data can be a scalar, vector, matrix, or higher-dimensional array with a fixed [`numpy.dtype`](https://numpy.org/doc/stable/reference/arrays.dtypes.html) and shape. An observable value can be considered as a time series that only stores its most recent value.

## Time Series

A [time series](python/tradingflow/series.py) stores a sequence of uniformly typed elements indexed by strictly increasing `np.datetime64` timestamps, supporting integer indexing, slicing, timestamp lookups, and amortized O(1) appends. Any observable value can be *materialized* into a time series.

An element in a time series can be a scalar, vector, matrix or higher-dimensional array with a fixed [`numpy.dtype`](https://numpy.org/doc/stable/reference/arrays.dtypes.html) and shape. Elements in a time series are internally stored in a single [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). This design is simple but not as flexible as Pandas-style data frames, which can contain columns of different types. To support such data, multiple time series must be created for different types.

## Sources

A [source](python/tradingflow/source.py) is used to generate observable values via asynchronous inputs from external data sources. A source must implement its `subscribe()` method, which returns two asynchronous iterators: one for historical `(timestamp, value)` tuples and one for real-time `value` updates. They should generate two complementary, non-overlapping segments of the same time series, split at some instant during the execution of `subscribe()`.

Sources are typically raw market data or pre-computed factors. Examples include:

- **Order flows** (e.g. transactions containing instrument name, price, quantity, etc.)
- **Snapshot prices** (e.g. arrays of all instrument prices updated at the end of each trading day)
- **Financial reports** (e.g. arrays of all balance sheet fields updated when a new financial report is released)

## Operators

An [operator](python/tradingflow/operator.py) is used to generate observable values via computations on other observable values or time series. An operator must implement its `compute()` method, which takes the current timestamp, a tuple of inputs and an optional mutable hidden state, and returns the updated output value. The `compute()` method is called to generate a new output from inputs when any of them is updated.

Operators are the reusable building blocks of trading strategies. Examples include:

- [**Technical indicators**](python/tradingflow/operators/indicators/) (e.g. 20-day moving average of instrument prices)
- [**Model predictions**](python/tradingflow/operators/predictors/) (e.g. from a regression model predicting future instrument returns, periodically retrained on historical data)
- [**Target positions**](python/tradingflow/operators/portfolios/) (e.g. periodically recomputed by mean-variance portfolio optimization on some forecasted returns and covariances)
- **Trading signals** (e.g. differences between target and actual positions)
- [**Performance metrics**](python/tradingflow/operators/metrics/) (e.g. cumulative returns calculated from past positions)

## Scenarios

A [scenario](python/tradingflow/scenario.py) is a collection of time series, each associated with either a source or an operator along with its input time series. Time series dependencies must be acyclic. It provides a `run()` method which consumes all source streams, coalesces source events (which are only required to have non-decreasing timestamps) so that update timestamps are strictly increasing, and for each event updates all affected downstream time series.
