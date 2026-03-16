"""Time series operators.

This package contains all concrete operator implementations, organized into
sub-packages by category:

* **operators** (this entry point): [`Apply`][tradingflow.operators.Apply], arithmetic factories
  ([`add`][tradingflow.operators.add], [`subtract`][tradingflow.operators.subtract], [`multiply`][tradingflow.operators.multiply], [`divide`][tradingflow.operators.divide],
  [`negate`][tradingflow.operators.negate]), [`Concat`][tradingflow.operators.Concat], [`Filter`][tradingflow.operators.Filter], [`Stack`][tradingflow.operators.Stack],
  [`Where`][tradingflow.operators.Where], and [`Select`][tradingflow.operators.Select].
* **operators.filters**: Rolling statistical filters and technical indicators.
* **operators.metrics**: Performance metrics (average return, Sharpe ratio).
* **operators.portfolios**: Portfolio construction methods.
* **operators.predictors**: Prediction models with rolling retraining.
* **operators.signals**: Trading signal generators (placeholder).
* **operators.simulators**: Trading simulators.

[`Filter`][tradingflow.operators.Filter] drops entire elements when a scalar predicate returns
`False`.  [`Where`][tradingflow.operators.Where] replaces individual array elements with a fill
value (element-wise conditional, analogous to `np.where`).
"""

from .apply import Apply, add, divide, map, multiply, negate, subtract
from .concat import Concat
from .filter import Filter
from .stack import Stack
from .select import Select, select
from .where import Where

__all__ = [
    "Apply",
    "Concat",
    "Filter",
    "Select",
    "Stack",
    "Where",
    "add",
    "divide",
    "map",
    "multiply",
    "negate",
    "select",
    "subtract",
]
