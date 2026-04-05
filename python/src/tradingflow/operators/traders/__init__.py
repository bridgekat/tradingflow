"""Trading simulation operators.

Trader operators convert ideal position weights into actual trades,
tracking cash, share holdings, and transaction costs.

All operators in this module are [`Operator`][tradingflow.Operator]
subclasses whose [`compute`][tradingflow.Operator.compute] method runs in
Python.

- [`SimpleTrader`][tradingflow.operators.traders.SimpleTrader] -- abstract
  base handling dividend adjustment, trade execution, fee deduction, and
  portfolio valuation under a simplified market model.
- [`Benchmark`][tradingflow.operators.traders.Benchmark] -- frictionless
  ideal portfolio replication (no transaction costs or lot rounding).

## Sub-modules

- [`simple`][tradingflow.operators.traders.simple] -- concrete
  simple-trader implementations.
"""

from . import simple
from .benchmark import Benchmark
from .simple_trader import SimpleTrader, SimpleTraderState

__all__ = [
    "simple",
    "Benchmark",
    "SimpleTrader",
    "SimpleTraderState",
]
