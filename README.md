**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. It is currently implemented in Python with NumPy.

# Features

- **Composable modules:** almost everything is a time series or a time series operator.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate AI code exploration and generation. When using AI coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

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

A [scenario](src/scenario.py) is a collection of source and derived series along with their dependencies, which must be acyclic.

## Sources and Runtime

Each [source](src/source.py) owns exactly one source series and emits values through an async stream.  
`Scenario.run()` consumes all source streams, appends updates, coalesces updates at identical timestamps, and incrementally updates affected downstream operators.

### CSV Source Example

```python
import asyncio
import numpy as np
from src import CSVSource, Scenario, Series

prices = Series((), np.dtype(np.float64))
source = CSVSource(
    "prices.csv",
    prices,
    timestamp_col="ts",
    value_cols=("close",),
)

scenario = Scenario(sources=(source,))
asyncio.run(scenario.run())
```

### Array Bundle / Pickle Source Example

```python
import asyncio
import numpy as np
from src import ArrayBundleSource, Scenario, Series

returns = Series((), np.dtype(np.float64))
source = ArrayBundleSource.from_pickle("returns.pkl", returns)

scenario = Scenario(sources=(source,))
asyncio.run(scenario.run())
```

### Realtime Async Source Example

```python
import asyncio
import numpy as np
from src import AsyncCallableSource, Scenario, Series

ticks = Series((), np.dtype(np.float64))

async def tick_stream():
    for value in (1.0, 2.0, 3.0):
        yield value

source = AsyncCallableSource(ticks, tick_stream)
scenario = Scenario(sources=(source,))
asyncio.run(scenario.run())
```

### Financial Report Source Example

```python
import asyncio
from src import Scenario
from src.ops import select_fields
from src.sources.eastmoney.history import FinancialReportCSVSource

income_source = FinancialReportCSVSource(
    "000001_income_statement_raw.csv",
    kind="income_statement",
)

# Pick canonical fields by vector index (for example, net profit).
profit_idx = income_source.schema.field_index["income_statement.profit"]
profit_series = select_fields(income_source.series, (profit_idx,))

scenario = Scenario(
    sources=(income_source,),
    operators=(profit_series,),
)
asyncio.run(scenario.run())
```

### Daily Market Snapshot Source Example

```python
import asyncio
from src import Scenario
from src.ops import select_fields
from src.sources.eastmoney.history import DailyMarketSnapshotCSVSource

price_source = DailyMarketSnapshotCSVSource("000001_daily_price_raw.csv")
close_idx = price_source.schema.field_index["close"]
close_series = select_fields(price_source.series, (close_idx,))

scenario = Scenario(
    sources=(price_source,),
    operators=(close_series,),
)
asyncio.run(scenario.run())
```

## Storage Policies (TODO)

Based on the scenario demand, a time series may admit one of the following storage policies:

- **Last:** only the most recent value is stored. Only available if historical values are never accessed.
- **Window:** all values are stored but oldest entries are eventually removed. Only available if all accesses are within a fixed time window.
- **Full:** all values are stored.

All time series are immutable: a value cannot be modified once it is generated.
