"""Concrete simple-trader implementations.

- [`RandomTrader`][tradingflow.operators.traders.simple.RandomTrader]
  -- rounds ideal position weights to the nearest lot.
"""

from .random_trader import RandomTrader

__all__ = ["RandomTrader"]
