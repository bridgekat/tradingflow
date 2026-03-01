**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. It is currently implemented in Python with NumPy.

# Features

- **Unified data model:** almost everything is a time series or a time series operator.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. Instruct AI agents to start every task by reading [AGENTS.md](AGENTS.md).

# Core Concepts

## Time Series

A [time series](src/series.py) is a sequence of uniformly typed values indexed by strictly increasing `np.datetime64` timestamps.

### Source Series

A source series is simply a time series of raw market data.

Examples include:

- **Order flows**
- **Snapshot prices**
- **Financial reports**
- **Order execution status**

### Derived Series

A derived series is a time series computed from zero or more time series by a [fixed algorithm](src/operator.py), either in real-time or periodically. The input series interface assumes "as-of" queries, ensuring causality of computation (i.e. the output value at a given timestamp only depends on input values from the past and present, not the future).

Examples include:

- **Summarized market data** (e.g. arrays of all instrument prices updated at fixed intervals)
- **Formulaic factors** (e.g. 20-day moving average of instrument prices)
- **Model states and forecasts** (e.g. from a regression model predicting future instrument returns, periodically retrained on historical data)
- **Target positions** (e.g. periodically recomputed by mean-variance portfolio optimization on some forecasted returns and covariances)
- **Actual positions** (e.g. calculated from order execution history)
- **Trading signals** (e.g. differences between target and actual positions)
- **Performance metrics** (e.g. cumulative returns calculated from past positions)

## Scenarios

A scenario is a collection of source and derived series along with their dependencies, which must be acyclic.

## Events

An event is an update to zero or more source series at a specific timestamp. Events are processed in timestamp order. Each event triggers incremental updates to all affected derived series in the scenario.

## Storage Policies (TODO)

Based on the scenario demand, a (source or derived) series may admit one of the following storage policies:

- **Last:** only the most recent value is stored. Only available if historical values are never accessed.
- **Window:** all values are stored but oldest entries are eventually removed. Only available if all accesses are within a fixed time window.
- **Full:** all values are stored.

All series are immutable: a value cannot be modified once it is generated.
