"""Concrete simple-trader implementations.

- [`RandomTrader`][tradingflow.operators.traders.simple.random_trader.RandomTrader] —
  samples `portfolio_size` stocks (without replacement) weighted by the
  soft position scores, and equal-weights them.
"""

from .random_trader import RandomTrader

__all__ = ["RandomTrader"]
