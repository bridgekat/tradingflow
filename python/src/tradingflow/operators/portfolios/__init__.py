"""Portfolio construction operators.

Portfolio operators convert predicted returns (and optionally covariance)
into theoretical position weights.

All operators in this module are [`Operator`][tradingflow.Operator]
subclasses whose [`compute`][tradingflow.Operator.compute] method runs in
Python.

- [`MeanPortfolio`][tradingflow.operators.portfolios.MeanPortfolio] --
  abstract base taking predicted returns only.
- [`MeanVariancePortfolio`][tradingflow.operators.portfolios.MeanVariancePortfolio] --
  abstract base taking predicted returns and covariance.

## Sub-modules

- [`mean`][tradingflow.operators.portfolios.mean] -- concrete
  mean-portfolio implementations.
- [`mean_variance`][tradingflow.operators.portfolios.mean_variance] -- concrete
  mean-variance portfolio implementations.
"""

from . import mean
from . import mean_variance
from .mean_portfolio import MeanPortfolio, MeanPortfolioState
from .mean_variance_portfolio import MeanVariancePortfolio, MeanVariancePortfolioState

__all__ = [
    "mean",
    "mean_variance",
    "MeanPortfolio",
    "MeanPortfolioState",
    "MeanVariancePortfolio",
    "MeanVariancePortfolioState",
]
