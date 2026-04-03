"""Stock-specific operators.

All operators in this module are [`NativeOperator`][tradingflow.NativeOperator]
subclasses dispatched entirely to Rust.

- [`ForwardAdjust`][tradingflow.operators.stocks.ForwardAdjust] -- forward price
  adjustment for corporate actions (dividends and share splits)
"""

from .forward_adjust import ForwardAdjust

__all__ = [
    "ForwardAdjust",
]
