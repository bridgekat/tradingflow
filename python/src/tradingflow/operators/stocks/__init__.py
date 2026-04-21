"""Stock-specific operators.

Small catalogue of operators that only really make sense for equity
data — the kind of transformations you'd write by hand once for each
dataset and then find yourself copy-pasting into every new project.
All are [`NativeOperator`][tradingflow.operator.NativeOperator] subclasses
dispatched entirely to Rust.

- [`ForwardAdjust`][tradingflow.operators.stocks.forward_adjust.ForwardAdjust] —
  adjust historical prices *forward* for corporate actions (cash
  dividends and share splits), so that returns computed from adjusted
  prices are directly comparable across event boundaries.  Takes a
  price input and a dividend input, and uses message-passing
  semantics internally (updates its cumulative adjustment factor only
  when the dividend input fires, and emits an adjusted price only
  when the price input fires).
- [`Annualize`][tradingflow.operators.stocks.annualize.Annualize] — convert a
  year-to-date cumulative financial-report value (net income, revenue,
  cash flow, etc.) into an annualized quarterly value using days-based
  scaling.  Turns lumpy Q1 / Q2 / Q3 / Q4 disclosures into a smoother
  rolling-year-equivalent series.
"""

from .annualize import Annualize
from .forward_adjust import ForwardAdjust

__all__ = [
    "Annualize",
    "ForwardAdjust",
]
