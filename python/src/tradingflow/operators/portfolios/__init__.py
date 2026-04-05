"""Portfolio construction operators.

Portfolio operators convert predicted returns into theoretical position
weights.

All operators in this module are [`Operator`][tradingflow.Operator]
subclasses whose [`compute`][tradingflow.Operator.compute] method runs in
Python.

- [`MeanPortfolio`][tradingflow.operators.portfolios.MeanPortfolio] --
  abstract base that dispatches to a user-provided positions function.

## Sub-modules

- [`mean`][tradingflow.operators.portfolios.mean] -- concrete
  mean-portfolio implementations.
"""

from . import mean
from .mean_portfolio import MeanPortfolio, MeanPortfolioState

__all__ = [
    "mean",
    "MeanPortfolio",
    "MeanPortfolioState",
]
