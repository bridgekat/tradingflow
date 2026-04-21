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

- [`Pow`][tradingflow.operators.num.pow.Pow] — raise each element to a
  constant exponent.
- [`Scale`][tradingflow.operators.num.scale.Scale] — multiply by a constant.
- [`Shift`][tradingflow.operators.num.shift.Shift] — add a constant.
- [`Clamp`][tradingflow.operators.num.clamp.Clamp] — clip values into a
  given `[lo, hi]` range.
- [`Fillna`][tradingflow.operators.num.fillna.Fillna] — replace NaNs with a
  constant.
- [`ForwardFill`][tradingflow.operators.num.ffill.ForwardFill] — replace
  NaNs with the last non-NaN value seen so far (stateful).

## Sorting

- [`ArgSort`][tradingflow.operators.num.argsort.ArgSort] — indices that would
  sort the input array.
"""

from .argsort import ArgSort
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
)
from .clamp import Clamp
from .ffill import ForwardFill
from .fillna import Fillna
from .pow import Pow
from .scale import Scale
from .shift import Shift

__all__ = [
    "ArgSort",
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
    "Scale",
    "Shift",
    "Clamp",
    "ForwardFill",
    "Fillna",
]
