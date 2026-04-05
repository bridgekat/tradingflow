"""Clock-driven financial metrics operators.

Each operator takes a scalar Array input and produces a scalar Array
output.  Intended to be triggered by a clock source so that each tick
represents one period (e.g. monthly).  All metrics are since-inception
(not rolling).

- [`CompoundReturn`][tradingflow.operators.metrics.CompoundReturn]
  -- `(current / first)^(1/n) - 1`
- [`AverageReturn`][tradingflow.operators.metrics.AverageReturn]
  -- mean of period returns
- [`Volatility`][tradingflow.operators.metrics.Volatility]
  -- population standard deviation of period returns
- [`SharpeRatio`][tradingflow.operators.metrics.SharpeRatio]
  -- mean / std of period returns
- [`Drawdown`][tradingflow.operators.metrics.Drawdown]
  -- `(current - running_max) / running_max`
"""

from .compound_return import CompoundReturn
from .average_return import AverageReturn
from .volatility import Volatility
from .sharpe_ratio import SharpeRatio
from .drawdown import Drawdown

__all__ = [
    "CompoundReturn",
    "AverageReturn",
    "Volatility",
    "SharpeRatio",
    "Drawdown",
]
