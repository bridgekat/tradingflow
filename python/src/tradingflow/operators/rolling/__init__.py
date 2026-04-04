"""Rolling window operators.

All operators in this module are [`NativeOperator`][tradingflow.NativeOperator]
subclasses operating on Series inputs with float dtypes, dispatched entirely
to Rust. Each operator outputs an Array (not a Series); use
[`Record`][tradingflow.operators.Record] to accumulate output into a Series.

Rolling operators accept a *window* parameter that is either an `int`
(count-based: last N elements) or a `numpy.timedelta64` (time-delta-based:
all elements within the given duration of the most recent timestamp).

- [`RollingSum`][tradingflow.operators.rolling.RollingSum],
  [`RollingMean`][tradingflow.operators.rolling.RollingMean],
  [`RollingVariance`][tradingflow.operators.rolling.RollingVariance],
  [`RollingCovariance`][tradingflow.operators.rolling.RollingCovariance]
- [`EMA`][tradingflow.operators.rolling.EMA] -- window-normalized exponential
  moving average
- [`ForwardFill`][tradingflow.operators.rolling.ForwardFill] -- forward-fill
  NaN values
"""

from .sum import RollingSum
from .mean import RollingMean
from .variance import RollingVariance
from .covariance import RollingCovariance
from .ema import EMA
from .ffill import ForwardFill

__all__ = [
    "RollingSum",
    "RollingMean",
    "RollingVariance",
    "RollingCovariance",
    "EMA",
    "ForwardFill",
]
