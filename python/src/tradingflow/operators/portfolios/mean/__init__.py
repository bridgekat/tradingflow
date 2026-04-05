"""Concrete mean-portfolio implementations.

- [`Proportional`][tradingflow.operators.portfolios.mean.Proportional]
  -- weights proportional to positive predicted returns.
- [`Softmax`][tradingflow.operators.portfolios.mean.Softmax]
  -- softmax-weighted by predicted returns with temperature control.
- [`RankEqual`][tradingflow.operators.portfolios.mean.RankEqual]
  -- equal weights to the top fraction of stocks.
- [`RankLinear`][tradingflow.operators.portfolios.mean.RankLinear]
  -- linearly decreasing weights to the top fraction of stocks.
"""

from .proportional import Proportional
from .rank_equal import RankEqual
from .rank_linear import RankLinear
from .softmax import Softmax

__all__ = [
    "Proportional",
    "RankEqual",
    "RankLinear",
    "Softmax",
]
