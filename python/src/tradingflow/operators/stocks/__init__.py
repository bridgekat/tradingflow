"""Stock-specific operators.

All operators in this module are [`NativeOperator`][tradingflow.NativeOperator]
subclasses dispatched entirely to Rust.

- [`Annualize`][tradingflow.operators.stocks.Annualize] -- convert year-to-date
  financial report values into annualised quarterly values using days-based scaling
- [`ForwardAdjust`][tradingflow.operators.stocks.ForwardAdjust] -- forward price
  adjustment for corporate actions (dividends and share splits)
"""

from .annualize import Annualize
from .forward_adjust import ForwardAdjust

__all__ = [
    "Annualize",
    "ForwardAdjust",
]
