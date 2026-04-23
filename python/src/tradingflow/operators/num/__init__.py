"""Element-wise numeric operators.

Stateless arithmetic and math applied element-by-element to array
nodes — the low-level building blocks you reach for when composing a
formulaic factor or a small derived signal.  All operators in this
module are [`NativeOperator`][tradingflow.operator.NativeOperator] subclasses
dispatched entirely to Rust, so they run at roughly NumPy speed with
no per-element Python overhead.

Inputs and outputs are always `Array<T>` nodes of the same shape.
Operators that require a specific dtype class (e.g. floating-point
for `Log`) check that at construction time, so mistakes surface before
the event loop starts.

## Arithmetic

- Binary: [`Add`][tradingflow.operators.num.arithmetic.Add],
  [`Subtract`][tradingflow.operators.num.arithmetic.Subtract],
  [`Multiply`][tradingflow.operators.num.arithmetic.Multiply],
  [`Divide`][tradingflow.operators.num.arithmetic.Divide].
- Unary: [`Negate`][tradingflow.operators.num.arithmetic.Negate].

## Float unary math

- Log / exp: [`Log`][tradingflow.operators.num.arithmetic.Log],
  [`Log2`][tradingflow.operators.num.arithmetic.Log2],
  [`Log10`][tradingflow.operators.num.arithmetic.Log10],
  [`Exp`][tradingflow.operators.num.arithmetic.Exp],
  [`Exp2`][tradingflow.operators.num.arithmetic.Exp2],
  [`Sqrt`][tradingflow.operators.num.arithmetic.Sqrt].
- Rounding / reciprocal: [`Ceil`][tradingflow.operators.num.arithmetic.Ceil],
  [`Floor`][tradingflow.operators.num.arithmetic.Floor],
  [`Round`][tradingflow.operators.num.arithmetic.Round],
  [`Recip`][tradingflow.operators.num.arithmetic.Recip].

## Signed unary math

- [`Abs`][tradingflow.operators.num.arithmetic.Abs],
  [`Sign`][tradingflow.operators.num.arithmetic.Sign].

## Float binary math

- [`Min`][tradingflow.operators.num.arithmetic.Min],
  [`Max`][tradingflow.operators.num.arithmetic.Max].

## Parameterized unary

These take a constant parameter at construction time and apply it
element-wise at compute time:

- [`Pow`][tradingflow.operators.num.arithmetic.Pow] — raise each element
  to a constant exponent.
- [`Clamp`][tradingflow.operators.num.clamp.Clamp] — clip values into a
  given `[lo, hi]` range.
- [`Fillna`][tradingflow.operators.num.fillna.Fillna] — replace NaNs with a
  constant.
- [`ForwardFill`][tradingflow.operators.num.ffill.ForwardFill] — replace
  NaNs with the last non-NaN value seen so far (stateful).

## Cross-tick (stateful)

Single-step stateful operators that remember the previous tick.  Output
is `NaN` on the first tick.

- [`Diff`][tradingflow.operators.num.diff.Diff] — first difference:
  `a_t - a_{t-1}`.
- [`PctChange`][tradingflow.operators.num.pct_change.PctChange] — linear
  return: `a_t / a_{t-1} - 1`.

Use `PctChange` for linear returns and `Log -> Diff` for log returns.

## Distribution shaping

Cross-sectional rank statistics that sort and handle NaN internally:
non-NaN entries are ranked ascending (the denominator is ``n_valid``,
not ``n``) and NaN inputs propagate to NaN outputs, so downstream
``np.isfinite`` masks still filter missing entries.  1-D float inputs
only.

- [`Gaussianize`][tradingflow.operators.num.gaussianize.Gaussianize] —
  cross-sectional rank-to-Gaussian transform: map each non-NaN element
  to ``Φ⁻¹((rank + 0.5) / n_valid)``.
- [`Percentile`][tradingflow.operators.num.percentile.Percentile] —
  cross-sectional rank-to-percentile transform: map each non-NaN
  element to ``(rank + 0.5) / n_valid ∈ (0, 1)``.  Same sort and NaN
  logic as ``Gaussianize``, just without the ``Φ⁻¹`` step.
"""

from .arithmetic import (
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Log,
    Log2,
    Log10,
    Exp,
    Exp2,
    Sqrt,
    Ceil,
    Floor,
    Round,
    Recip,
    Abs,
    Sign,
    Min,
    Max,
    Pow,
)
from .clamp import Clamp
from .diff import Diff
from .ffill import ForwardFill
from .fillna import Fillna
from .gaussianize import Gaussianize
from .pct_change import PctChange
from .percentile import Percentile

__all__ = [
    "Gaussianize",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Negate",
    "Log",
    "Log2",
    "Log10",
    "Exp",
    "Exp2",
    "Sqrt",
    "Ceil",
    "Floor",
    "Round",
    "Recip",
    "Abs",
    "Sign",
    "Min",
    "Max",
    "Pow",
    "Clamp",
    "Diff",
    "ForwardFill",
    "Fillna",
    "PctChange",
    "Percentile",
]
