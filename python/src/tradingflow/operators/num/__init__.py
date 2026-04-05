"""Element-wise numeric operators.

All operators in this module are [`NativeOperator`][tradingflow.NativeOperator]
subclasses -- stateless, element-wise arithmetic and math on `Array<T>`,
dispatched entirely to Rust.

## Arithmetic

- [`Add`][tradingflow.operators.num.Add],
  [`Subtract`][tradingflow.operators.num.Subtract],
  [`Multiply`][tradingflow.operators.num.Multiply],
  [`Divide`][tradingflow.operators.num.Divide] -- binary
- [`Negate`][tradingflow.operators.num.Negate] -- unary

## Float unary math

- [`Log`][tradingflow.operators.num.Log],
  [`Log2`][tradingflow.operators.num.Log2],
  [`Log10`][tradingflow.operators.num.Log10],
  [`Exp`][tradingflow.operators.num.Exp],
  [`Exp2`][tradingflow.operators.num.Exp2],
  [`Sqrt`][tradingflow.operators.num.Sqrt]
- [`Ceil`][tradingflow.operators.num.Ceil],
  [`Floor`][tradingflow.operators.num.Floor],
  [`Round`][tradingflow.operators.num.Round],
  [`Recip`][tradingflow.operators.num.Recip]

## Signed unary math

- [`Abs`][tradingflow.operators.num.Abs],
  [`Sign`][tradingflow.operators.num.Sign]

## Float binary math

- [`Min`][tradingflow.operators.num.Min],
  [`Max`][tradingflow.operators.num.Max]

## Parameterized unary

- [`Pow`][tradingflow.operators.num.Pow],
  [`Scale`][tradingflow.operators.num.Scale],
  [`Shift`][tradingflow.operators.num.Shift],
  [`Clamp`][tradingflow.operators.num.Clamp],
  [`Fillna`][tradingflow.operators.num.Fillna],
  [`ForwardFill`][tradingflow.operators.num.ForwardFill]
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
)
from .clamp import Clamp
from .ffill import ForwardFill
from .fillna import Fillna
from .pow import Pow
from .scale import Scale
from .shift import Shift

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
    "Min",
    "Max",
    "Pow",
    "Scale",
    "Shift",
    "Clamp",
    "ForwardFill",
    "Fillna",
]
