"""Built-in operators — the reusable building blocks of every strategy.

See the root [`tradingflow`][tradingflow] page for the conceptual
overview (what an operator is, notification semantics, etc.) and
[`Operator`][tradingflow.operator.Operator] /
[`NativeOperator`][tradingflow.operator.NativeOperator] for the two
implementation tiers.

This module groups operators by what they *do*.  The sub-modules below
cover the heavy-duty categories; the directly-exported names at the
top of this file cover the glue pieces you need in nearly every graph.

## Structural operators (reshape / reroute / combine)

Plumbing for moving data between nodes without mathematical
transformation.

- [`Id`][tradingflow.operators.id.Id] — identity passthrough.  Useful for
  alias handles or for benchmarking.
- [`Cast`][tradingflow.operators.cast.Cast] — element-wise dtype
  conversion (e.g. `int32` → `float64`).
- [`Const`][tradingflow.operators.const.Const] — zero-input node that always
  produces the same constant array.
- [`Select`][tradingflow.operators.select.Select] — pick specific elements
  out of an array by flat index (e.g. extract the "close" column from
  a prices row).
- [`Concat`][tradingflow.operators.concat.Concat] — concatenate N arrays
  along an existing axis.
- [`Stack`][tradingflow.operators.stack.Stack] — stack N arrays along a new
  axis.

### Synchronized (message-passing) variants

[`ConcatSync`][tradingflow.operators.concat.ConcatSync] and
[`StackSync`][tradingflow.operators.stack.StackSync] are the float-only
message-passing counterparts of `Concat` / `Stack` (non-produced input
slots are filled with `NaN`).

## Series operators (array ↔ series conversion)

Bridges between snapshot arrays and their full-history counterparts.

- [`Record`][tradingflow.operators.record.Record] — accumulate every `Array`
  value into a `Series`.  This is how you materialize histories for
  end-of-run inspection or as input to rolling / predictor operators.
- [`Last`][tradingflow.operators.last.Last] — expose a `Series`'s most
  recent value as an `Array`.
- [`Lag`][tradingflow.operators.lag.Lag] — output the value from N steps
  ago.

## Custom-function operators (Python)

For quick experiments without writing a full operator subclass.

- [`Map`][tradingflow.operators.map.Map] — apply a Python callable to
  every upstream array value.
- [`Apply`][tradingflow.operators.apply.Apply] — apply a Python callable
  that takes multiple array inputs.
- [`Filter`][tradingflow.operators.filter.Filter] — predicate-gated
  passthrough; when the predicate returns `False`, the element is
  *dropped* and downstream nodes are not notified.
- [`Where`][tradingflow.operators.where.Where] — element-wise conditional
  replacement; always produces output (never halts propagation).

## Scheduling

- [`Clocked`][tradingflow.operators.clocked.Clocked] — wrap any operator so
  that it only fires when a clock input ticks.  Convenient for turning
  a purely data-driven operator into a periodic one without modifying
  its implementation.

## Sub-modules

| Sub-module | Purpose |
|------------|---------|
| [`num`][tradingflow.operators.num] | element-wise arithmetic and math on arrays |
| [`rolling`][tradingflow.operators.rolling] | rolling windows over series (mean, variance, covariance, EMA) |
| [`predictors`][tradingflow.operators.predictors] | cross-sectional return and covariance predictors |
| [`portfolios`][tradingflow.operators.portfolios] | portfolio construction (mean, mean-variance, min-variance) |
| [`traders`][tradingflow.operators.traders] | simulated execution with transaction costs |
| [`metrics`][tradingflow.operators.metrics] | clock-driven performance metrics (Sharpe, drawdown, IC, ...) |
| [`stocks`][tradingflow.operators.stocks] | stock-specific helpers (forward adjustment, annualization) |
"""

from . import metrics, num, portfolios, predictors, rolling, stocks, traders

from .apply import Apply
from .cast import Cast
from .clocked import Clocked
from .concat import Concat, ConcatSync
from .const import Const
from .filter import Filter
from .id import Id
from .lag import Lag
from .map import Map
from .last import Last
from .record import Record
from .select import Select
from .stack import Stack, StackSync
from .where import Where

__all__ = [
    "metrics",
    "num",
    "portfolios",
    "predictors",
    "rolling",
    "stocks",
    "traders",
    "Apply",
    "Cast",
    "Clocked",
    "Const",
    "Concat",
    "ConcatSync",
    "Stack",
    "StackSync",
    "Select",
    "Id",
    "Map",
    "Record",
    "Last",
    "Lag",
    "Filter",
    "Where",
]
