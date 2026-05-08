"""Concrete mean-portfolio implementations.

- [`Proportional`][tradingflow.operators.portfolios.mean.proportional.Proportional]
  -weights proportional to positive predicted returns.
- [`Softmax`][tradingflow.operators.portfolios.mean.softmax.Softmax]
  -softmax-weighted by predicted returns with temperature control.
- [`RankEqual`][tradingflow.operators.portfolios.mean.rank_equal.RankEqual]
  -equal weights to the top fraction of stocks.
- [`RankLinear`][tradingflow.operators.portfolios.mean.rank_linear.RankLinear]
  -linearly decreasing weights to the top fraction of stocks.
- [`RankBucket`][tradingflow.operators.portfolios.mean.rank_bucket.RankBucket]
  -equal weights within a percentile slice of a feature, designed
  for decile / quantile portfolio analysis.
"""

from .proportional import Proportional
from .rank_bucket import RankBucket
from .rank_equal import RankEqual
from .rank_linear import RankLinear
from .softmax import Softmax

__all__ = [
    "Proportional",
    "RankBucket",
    "RankEqual",
    "RankLinear",
    "Softmax",
]
