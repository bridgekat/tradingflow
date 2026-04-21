"""Rolling-window operators over series inputs.

Rolling operators take one or more `Series` inputs with float dtype
and produce a single `Array` output (the latest rolling statistic).
If you want the full history of that statistic, wrap the output with
[`Record`][tradingflow.operators.record.Record].

All operators in this module are
[`NativeOperator`][tradingflow.operator.NativeOperator] subclasses dispatched
entirely to Rust.  They maintain incremental state internally, so each
new series element costs only O(1) work regardless of window size.

## Window parameter

Every rolling operator takes a `window` parameter that may be either:

- an `int` — a **count-based** window covering the last N elements, or
- a `numpy.timedelta64` — a **time-delta-based** window covering every
  element whose timestamp is within the given duration of the most
  recent one.

The two modes differ in how warm-up is handled:

- **Count-based** — the operator is *silent* until it has seen at
  least N elements.  This matches the `pandas` `rolling().mean()`
  convention (the first N-1 outputs would be `NaN`, but here no output
  is emitted at all).
- **Time-delta-based** — the operator emits as soon as the first
  element arrives, computing over whatever is already in the window.
  Short warm-up periods therefore still yield output, which may be
  noisier than steady-state values.

## Operators

- [`RollingSum`][tradingflow.operators.rolling.sum.RollingSum] — sum over
  the window.
- [`RollingMean`][tradingflow.operators.rolling.mean.RollingMean] —
  arithmetic mean over the window.
- [`RollingVariance`][tradingflow.operators.rolling.variance.RollingVariance] —
  population variance over the window.
- [`RollingCovariance`][tradingflow.operators.rolling.covariance.RollingCovariance] —
  covariance between two series over the window.
- [`EMA`][tradingflow.operators.rolling.ema.EMA] — window-normalized
  exponential moving average.  Unlike a pure EMA, this version
  converges to the true mean as its buffer fills, which tends to
  behave better during warm-up.
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
