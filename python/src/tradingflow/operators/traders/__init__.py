"""Trading simulation operators.

Trader operators close the loop between strategy and market: they take
*target* portfolio weights (from
[`portfolios`][tradingflow.operators.portfolios]) and simulate the
execution of orders needed to reach those targets, tracking cash,
share holdings, dividends, and transaction costs along the way.

The output of a trader is typically a single-scalar Series of
portfolio-value-over-time — exactly what
[`metrics`][tradingflow.operators.metrics] operators like
`SharpeRatio` and `Drawdown` expect to consume.

## Abstract bases

- [`SimpleTrader`][tradingflow.operators.traders.simple_trader.SimpleTrader] — the
  realistic path.  Handles dividend reinvestment, trade execution at
  the next available price, fee and slippage deduction, lot
  rounding, and NAV valuation under a simplified market model.
  Subclass it to customize any of those behaviors.
- [`Benchmark`][tradingflow.operators.traders.benchmark.Benchmark] — a
  frictionless reference.  Replicates the ideal portfolio exactly
  (no transaction costs, no lot rounding, instant fills).  Useful as
  the "best case" yardstick to measure how much a realistic trader
  gives up to frictions.

## Sub-modules

- [`simple`][tradingflow.operators.traders.simple] — concrete
  simple-trader implementations (e.g. a random-trader used for
  testing).
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
