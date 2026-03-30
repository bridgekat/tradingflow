"""Built-in operator classes for the computation graph.

This module provides all shipped operators. They fall into two categories:

- **Native operators** -- [`NativeOperator`][tradingflow.NativeOperator] subclasses
  whose computation is dispatched entirely to Rust for performance.
- **Python operators** -- [`Operator`][tradingflow.Operator] subclasses whose
  `compute` method runs in Python (under the GIL).

## Arithmetic (binary, element-wise)

- [`Add`][tradingflow.operators.Add] -- `a + b`
- [`Subtract`][tradingflow.operators.Subtract] -- `a - b`
- [`Multiply`][tradingflow.operators.Multiply] -- `a * b`
- [`Divide`][tradingflow.operators.Divide] -- `a / b`

## Float unary math (element-wise)

- [`Log`][tradingflow.operators.Log], [`Log2`][tradingflow.operators.Log2],
  [`Log10`][tradingflow.operators.Log10] -- logarithms
- [`Exp`][tradingflow.operators.Exp], [`Exp2`][tradingflow.operators.Exp2] --
  exponentials
- [`Sqrt`][tradingflow.operators.Sqrt] -- square root
- [`Ceil`][tradingflow.operators.Ceil], [`Floor`][tradingflow.operators.Floor],
  [`Round`][tradingflow.operators.Round] -- rounding
- [`Recip`][tradingflow.operators.Recip] -- reciprocal (`1/a`)

## Signed unary math (element-wise)

- [`Negate`][tradingflow.operators.Negate] -- `-a`
- [`Abs`][tradingflow.operators.Abs] -- absolute value
- [`Sign`][tradingflow.operators.Sign] -- signum (-1, 0, or +1)

## Parameterized unary (element-wise, native)

- [`Pow`][tradingflow.operators.Pow] -- `a ** n`
- [`Scale`][tradingflow.operators.Scale] -- `a * c`
- [`Shift`][tradingflow.operators.Shift] -- `a + c`
- [`Clamp`][tradingflow.operators.Clamp] -- clamp to `[lo, hi]`
- [`Fillna`][tradingflow.operators.Fillna] -- replace NaN with a constant

## Float binary math (element-wise)

- [`Min`][tradingflow.operators.Min], [`Max`][tradingflow.operators.Max] --
  element-wise min/max (IEEE 754 NaN semantics)

## Structural operators (native)

- [`Cast`][tradingflow.operators.Cast] -- element-wise dtype conversion
- [`Const`][tradingflow.operators.Const] -- zero-input constant array node
- [`Concat`][tradingflow.operators.Concat] -- concatenate N arrays along an
  existing axis
- [`Stack`][tradingflow.operators.Stack] -- stack N arrays along a new axis
- [`Select`][tradingflow.operators.Select] -- select elements by flat indices
- [`Record`][tradingflow.operators.Record] -- accumulate Array values into a Series
- [`Id`][tradingflow.operators.Id] -- identity passthrough
- [`Last`][tradingflow.operators.Last] -- extract most recent value from a Series
- [`Lag`][tradingflow.operators.Lag] -- output the value from N steps ago

## Rolling window operators (native, float dtypes only)

- [`RollingSum`][tradingflow.operators.RollingSum] -- rolling sum
- [`RollingMean`][tradingflow.operators.RollingMean] -- rolling mean
- [`RollingVariance`][tradingflow.operators.RollingVariance] -- rolling population
  variance
- [`RollingCovariance`][tradingflow.operators.RollingCovariance] -- rolling pairwise
  covariance matrix (1-D input -> 2-D output)
- [`EMA`][tradingflow.operators.EMA] -- window-normalized exponential moving average
  (specify one of `alpha`, `span`, or `half_life`)
- [`ForwardFill`][tradingflow.operators.ForwardFill] -- forward-fill NaN values

## Python operators

- [`Filter`][tradingflow.operators.Filter] -- predicate-gated passthrough; drops the
  entire element when the predicate returns `False`, halting downstream propagation
- [`Where`][tradingflow.operators.Where] -- element-wise conditional replacement;
  always produces output (never halts propagation)
"""

from .num import (
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
    Pow,
    Scale,
    Shift,
    Clamp,
    Fillna,
    Min,
    Max,
)
from .cast import Cast
from .concat import Concat
from .const import Const
from .filter import Filter
from .id import Id
from .lag import Lag
from .last import Last
from .record import Record
from .rolling import (
    EMA,
    ForwardFill,
    RollingCovariance,
    RollingMean,
    RollingSum,
    RollingVariance,
)
from .select import Select
from .stack import Stack
from .where import Where

__all__ = [
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
    "Pow",
    "Scale",
    "Shift",
    "Clamp",
    "Fillna",
    "Min",
    "Max",
    "Cast",
    "Const",
    "Concat",
    "Stack",
    "Select",
    "Record",
    "Id",
    "Last",
    "Lag",
    "RollingSum",
    "RollingMean",
    "RollingVariance",
    "RollingCovariance",
    "EMA",
    "ForwardFill",
    "Filter",
    "Where",
]
