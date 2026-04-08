[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. The core runtime is implemented in Rust; a Python wrapper is provided for ease of use with Python's data science ecosystem.

# Features

- **Composable modules:** trading strategies are computation graphs, whose nodes are either data sources or operators. Common sources and operators are provided out of the box, and new ones can be readily implemented in either Rust or Python.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. When using AI coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

# Examples

Examples require A-shares market data downloaded via the [a-shares-crawler](https://github.com/bridgekat/a-shares-crawler). Install optional dependencies with `pip install -e ".[examples]"`. Then run `python -m a_shares_crawler --help` for configuration and download instructions.

- [**Plotting daily prices**](python/examples/plot_daily_price.py) — loads daily price history, computes forward-adjusted prices, moving average and Bollinger Bands, and plots them.
- [**Plotting financial data**](python/examples/plot_financial_data.py) — loads equity structure, balance sheet, income statement and cash flow data, computes market cap and annualized financial metrics, and plots them.
- [**Plotting total market cap**](python/examples/plot_total_market_cap.py) — loads daily prices and equity structures for all stocks, computes per-stock circulating market cap, and plots the total across the entire market over time.
- [**Mean strategy backtesting**](python/examples/mean_strategy.py) — loads daily prices, equity structures, dividends, and financial reports for all stocks, computes cross-sectional features, periodically fits a linear regression to predict stock returns, selects a portfolio from the top-predicted stocks with rank-linear weights, simulates trading with transaction costs, and plots portfolio value, rolling Sharpe ratio and drawdown against a market-cap-weighted index.
- [**Mean-variance strategy backtesting**](python/examples/mean_variance_strategy.py) — extends the mean strategy with Ledoit-Wolf shrinkage covariance estimation and Markowitz mean-variance portfolio optimisation (via CVXPY), comparing multiple risk-aversion levels against the index.

# Core Concepts

## Arrays and Series

A [multi-dimensional array](src/array.rs) contains uniformly-typed [scalars](src/types.rs). It is backed by a flat `Vec<T>` scalar buffer, guaranteeing contiguous layout.

A [time series](src/series.rs) contains uniformly-shaped array elements. It is backed by a flat `Vec<T>` scalar buffer and a parallel `Vec<i64>` of non-decreasing timestamps (each representing nanoseconds since the UNIX epoch).

Most nodes in the computation graph hold either arrays or series as their output values. They can be converted back and forth by a pair of inverse operators, assuming series elements are pushed one-by-one:

- The built-in [record](src/operators/record.rs) operator records every historical value of its input node (array) into its output node (series).
- The built-in [last](src/operators/last.rs) operator only keeps the most recent value of its input node (series) in its output node (array).

## Sources

A [source](src/source.rs) feeds data into a node via asynchronous channels. A source must implement its `init()` method, which consumes the source and returns two channel receivers: one for historical `(timestamp, event)` tuples and one for real-time. They should generate two complementary, non-overlapping segments of the same data stream, split at some instant during the execution of `init()`.

Sources are typically raw market data or pre-computed factors. Examples include:

- **Order flows** (e.g. transactions containing instrument name, price, quantity, etc.)
- **Snapshot prices** (e.g. arrays of all instrument prices updated at the end of each trading day)
- **Financial reports** (e.g. arrays of all balance sheet fields updated when a new financial report is released)

## Operators

An [operator](src/operator.rs) reads from one or more input nodes and writes data into an output node. An operator must implement its `compute()` method, which takes the current state, references to input data, mutable reference to the output data, and returns whether its downstream nodes should be notified. The `compute()` method is called every time an upstream node notifies.

Operators are the reusable building blocks of trading strategies. Examples include:

- [**Technical indicators**](src/operators/rolling/) (e.g. 20-day moving average of instrument prices)
- [**Model predictions**](python/src/tradingflow/operators/predictors/) (e.g. from a regression model predicting future instrument returns, periodically retrained on historical data)
- [**Target positions**](python/src/tradingflow/operators/portfolios/) (e.g. periodically recomputed by mean-variance portfolio optimization on some forecasted returns and covariances)
- [**Trading simulators**](python/src/tradingflow/operators/traders/) (e.g. simulated execution of desired target positions with transaction costs, slippage, and other trading frictions)
- [**Performance metrics**](src/operators/metrics/) (e.g. Sharpe ratios calculated from past portfolio values)

## Scenarios

A [scenario](src/scenario/mod.rs) stores and runs the acyclic computation graph. Each node is associated with either a source or an operator.

The scenario provides a `run()` method which consumes all source streams (historical and real-time) and puts them in a queue, coalesces events at the same timestamp, and for each batch propagates updates through the graph in topological order. It makes sure that update timestamps are strictly increasing each batch.

## Notification Semantics

When a source emits or an operator produces output, its downstream operators are scheduled. Each downstream operator receives a `Notify` context that reports *which* of its inputs produced new output in the current flush cycle. This enables two complementary semantics:

### Time-series semantics

Each node's output array always holds a value (the most recent one written). An operator reads `inputs[i].value()` to get the latest snapshot, regardless of whether input `i` produced this cycle. This is the natural model for state that persists over time: prices, positions, factor values.

Most operators use time-series semantics and ignore the `Notify` context entirely — they simply recompute from the latest values whenever triggered.

### Message-passing semantics

A notification is treated as a *message* whose payload is the value written into the corresponding node. If an input did not produce this cycle, there is no message from it. Operators inspect `Notify` to distinguish "input updated" from "input is stale":

- [`Notify.produced()`](src/types.rs) returns the list of input positions that produced — for efficient O(n_messages) iteration over only the inputs that changed.
- [`Notify.input_produced()`](src/types.rs) returns a per-position boolean slice — for O(1) checks like `notify.input_produced()[i]`.

Message-passing semantics are useful when an operator must react differently depending on *which* inputs changed. For example, the [ForwardAdjust](src/operators/stocks/forward_adjust.rs) operator only updates its cumulative dividend factor when the dividend input produces, and only emits an adjusted price when the price input produces.

### Clock-triggerable operators

Operators may optionally be registered with a *clock* trigger instead of being triggered by their data inputs. A clock-triggered operator only runs when the clock fires, reading the latest values from its data inputs at that point.

Operators that rely on message-passing semantics must **not** be clock-triggered, since they would miss messages between clock ticks. The `Operator` trait provides an [`is_clock_triggerable()`](src/operator.rs) method (default `true`) that operators can override to return `false`, preventing accidental registration with a clock.
