"""Built-in operator factories.

Native operators return a [`NativeOperator`][tradingflow.NativeOperator]
descriptor dispatched to Rust. Python operators (`Filter`, `Where`) are
[`Operator`][tradingflow.Operator] subclasses whose `compute` runs via GIL.
"""

from .apply import (
    # Arithmetic
    add, subtract, multiply, divide, negate,
    # Float unary math
    log, log2, log10, exp, exp2, sqrt, ceil, floor, round, recip,
    # Signed unary math
    abs, sign,
    # Parameterized unary
    pow, scale, shift, clamp, nan_to_num,
    # Float binary math
    min, max,
)
from .cast import cast
from .concat import concat
from .const import const
from .filter import Filter
from .id import id
from .lag import lag
from .last import last
from .record import record
from .rolling import (
    ema,
    forward_fill,
    rolling_covariance,
    rolling_mean,
    rolling_sum,
    rolling_variance,
)
from .select import select
from .stack import stack
from .where import Where

__all__ = [
    "Filter",
    "Where",
    # Arithmetic
    "add", "subtract", "multiply", "divide", "negate",
    # Float unary math
    "log", "log2", "log10", "exp", "exp2", "sqrt",
    "ceil", "floor", "round", "recip",
    # Signed unary math
    "abs", "sign",
    # Parameterized unary
    "pow", "scale", "shift", "clamp", "nan_to_num",
    # Float binary math
    "min", "max",
    # Structural
    "cast", "const", "concat", "stack", "select", "record",
    "id", "last", "lag",
    # Rolling
    "rolling_sum", "rolling_mean", "rolling_variance", "rolling_covariance",
    "ema", "forward_fill",
]
