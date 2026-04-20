"""Built-in operator classes for the computation graph.

This module provides all shipped operators. They fall into two categories:

- **Native operators** -- [`NativeOperator`][tradingflow.NativeOperator] subclasses
  whose computation is dispatched entirely to Rust for performance.
- **Python operators** -- [`Operator`][tradingflow.Operator] subclasses whose
  [`compute`][tradingflow.Operator.compute] method runs in Python (under the GIL).

## Structural operators

- [`Cast`][tradingflow.operators.Cast] -- element-wise dtype conversion
- [`Const`][tradingflow.operators.Const] -- zero-input constant array node
- [`Concat`][tradingflow.operators.Concat] -- concatenate N arrays along an existing axis
- [`Stack`][tradingflow.operators.Stack] -- stack N arrays along a new axis
- [`NotifyConcat`][tradingflow.operators.NotifyConcat] -- notify-aware Concat (float-only):
  fills non-produced input slots with NaN (message-passing semantics)
- [`NotifyStack`][tradingflow.operators.NotifyStack] -- notify-aware Stack (float-only):
  fills non-produced input slots with NaN (message-passing semantics)
- [`Select`][tradingflow.operators.Select] -- select elements by flat indices
- [`Id`][tradingflow.operators.Id] -- identity passthrough
- [`Map`][tradingflow.operators.Map] -- applies a function to transform array values
- [`MapInplace`][tradingflow.operators.MapInplace] -- applies a function in place on input and output arrays
- [`Apply`][tradingflow.operators.Apply] -- applies a function to multiple input arrays

## Series operators

- [`Record`][tradingflow.operators.Record] -- accumulate Array values into a Series
- [`Last`][tradingflow.operators.Last] -- extract most recent value from a Series
- [`Lag`][tradingflow.operators.Lag] -- output the value from N steps ago

## Python operators

- [`Filter`][tradingflow.operators.Filter] -- predicate-gated passthrough; drops the
  entire element when the predicate returns `False`, halting downstream propagation
- [`Where`][tradingflow.operators.Where] -- element-wise conditional replacement;
  always produces output (never halts propagation)

## Sub-modules

- [`num`][tradingflow.operators.num] -- element-wise numeric operators
- [`portfolios`][tradingflow.operators.portfolios] -- portfolio construction operators
- [`predictors`][tradingflow.operators.predictors] -- cross-sectional return predictors
- [`rolling`][tradingflow.operators.rolling] -- rolling window operators
- [`metrics`][tradingflow.operators.metrics] -- clock-driven financial metrics
- [`stocks`][tradingflow.operators.stocks] -- stock-specific operators
- [`traders`][tradingflow.operators.traders] -- trading simulation operators
"""

from . import metrics, num, portfolios, predictors, rolling, stocks, traders

from .apply import Apply
from .cast import Cast
from .clocked import Clocked
from .concat import Concat
from .const import Const
from .filter import Filter
from .id import Id
from .lag import Lag
from .map import Map
from .last import Last
from .notify_concat import NotifyConcat
from .notify_stack import NotifyStack
from .record import Record
from .select import Select
from .stack import Stack
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
    "NotifyConcat",
    "NotifyStack",
    "Stack",
    "Select",
    "Id",
    "Map",
    "Record",
    "Last",
    "Lag",
    "Filter",
    "Where",
]
