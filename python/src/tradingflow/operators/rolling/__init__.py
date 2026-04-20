"""Rolling window operators.

All operators in this module are [`NativeOperator`][tradingflow.NativeOperator]
subclasses operating on Series inputs with float dtypes, dispatched entirely
to Rust. Each operator outputs an Array (not a Series); use
[`Record`][tradingflow.operators.Record] to accumulate output into a Series.

Rolling operators accept a *window* parameter that is either an `int`
(count-based: last N elements) or a `numpy.timedelta64` (time-delta-based:
all elements within the given duration of the most recent timestamp).
The two strategies differ in when the operator starts emitting output:

- **Count-based (`int`)** — output is produced only once the window is
  full (i.e. after the first N elements).  Before then the operator is
  silent.
- **Time-delta-based (`numpy.timedelta64`)** — output is produced as
  soon as at least one element is in the window.  Short warm-up
  periods therefore still yield output, computed over whatever
  elements have arrived so far.

- [`RollingSum`][tradingflow.operators.rolling.RollingSum],
  [`RollingMean`][tradingflow.operators.rolling.RollingMean],
  [`RollingVariance`][tradingflow.operators.rolling.RollingVariance],
  [`RollingCovariance`][tradingflow.operators.rolling.RollingCovariance]
- [`EMA`][tradingflow.operators.rolling.EMA] -- window-normalized exponential
  moving average
"""

from .sum import RollingSum
from .mean import RollingMean
from .variance import RollingVariance
from .covariance import RollingCovariance
from .ema import EMA

__all__ = [
    "RollingSum",
    "RollingMean",
    "RollingVariance",
    "RollingCovariance",
    "EMA",
]
