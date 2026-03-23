"""Built-in operator factories.

Native operators return a [`NativeOperator`][tradingflow.NativeOperator]
descriptor dispatched to Rust. Python operators (`Filter`, `Where`) are
[`Operator`][tradingflow.Operator] subclasses whose `compute` runs via GIL.
"""

from .apply import add, subtract, multiply, divide, negate
from .concat import concat
from .filter import Filter
from .record import record
from .select import select
from .stack import stack
from .where import Where

__all__ = [
    "Filter",
    "Where",
    "add", "subtract", "multiply", "divide", "negate",
    "concat", "stack", "select", "record",
]
