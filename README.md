**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. It is currently implemented in Python with NumPy.

# Features

- **Composable modules:** almost everything is a time series or a time series operator.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. Instruct AI agents to start every task by reading [AGENTS.md](AGENTS.md).

# Core Concepts

## Time Series

A [time series](src/series.py) is a sequence of uniformly typed elements indexed by strictly increasing `np.datetime64` timestamps, supporting integer indexing, slicing, timestamp lookups, and amortized O(1) appends.

### Data Formats

An element in a time series can be a scalar, vector, matrix or higher-dimensional array with a fixed [`numpy.dtype`](https://numpy.org/doc/stable/reference/arrays.dtypes.html) and shape. Elements in a time series are internally stored in a single [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html).

This design is simple but not as flexible as Pandas-style data frames, which can contain columns of different types. To support such data, multiple time series must be created for different types.

### Source Series

A source series is simply a time series of raw market data.

Examples include:

- **Order flows** (TODO)
- **Snapshot prices** (TODO)
- **Financial reports** (TODO)
- **Order execution status** (TODO)

### Derived Series

A derived series is a time series computed from zero or more time series by an [operator](src/operator.py). An operator is defined by its compute function, which generates the current value given all input series *up to the current timestamp*, preventing accidental use of future information.

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

An event is an update to zero or more source series at a specific timestamp. Events are generated with strictly increasing timestamps. Each event triggers incremental updates to all affected derived series in the scenario.

## Storage Policies (TODO)

Based on the scenario demand, a time series may admit one of the following storage policies:

- **Last:** only the most recent value is stored. Only available if historical values are never accessed.
- **Window:** all values are stored but oldest entries are eventually removed. Only available if all accesses are within a fixed time window.
- **Full:** all values are stored.

All time series are immutable: a value cannot be modified once it is generated.
