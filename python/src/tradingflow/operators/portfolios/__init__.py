"""Portfolio construction operators.

Portfolio operators turn predictions into target weights — the
recipe for *what* the strategy should hold, before any consideration
of execution, costs, or lot sizes.  They sit between predictors
(which output return / covariance forecasts) and traders (which
convert target weights into actual orders).

The three abstract bases differ by what information they consume:

- [`MeanPortfolio`][tradingflow.operators.portfolios.mean_portfolio.MeanPortfolio] —
  takes predicted returns only.  Simple heuristics (top-N equal
  weight, rank-linear, softmax) that don't need covariance
  information.
- [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio] —
  takes both predicted returns and a predicted covariance matrix.
  The classic Markowitz setup: maximize expected return for a given
  risk aversion.
- [`VariancePortfolio`][tradingflow.operators.portfolios.variance_portfolio.VariancePortfolio] —
  takes covariance only.  For pure risk-minimizing allocations like
  the Global Minimum Variance portfolio — useful as a research
  baseline for evaluating covariance estimators (see
  [`metrics.variance`][tradingflow.operators.metrics.variance]).

## Sub-modules

- [`mean`][tradingflow.operators.portfolios.mean] — concrete
  return-only builders (proportional, rank-equal, rank-linear,
  softmax).
- [`mean_variance`][tradingflow.operators.portfolios.mean_variance] —
  concrete Markowitz-style builders (CVXPY-based and a pure-Python
  SCS-backed fallback).
- [`variance`][tradingflow.operators.portfolios.variance] — concrete
  variance-only builders (analytic global minimum variance).
"""

from . import mean, mean_variance, variance
from .mean_portfolio import MeanPortfolio, MeanPortfolioState
from .mean_variance_portfolio import MeanVariancePortfolio, MeanVariancePortfolioState
from .variance_portfolio import VariancePortfolio, VariancePortfolioState

__all__ = [
    "mean",
    "mean_variance",
    "variance",
    "MeanPortfolio",
    "MeanPortfolioState",
    "MeanVariancePortfolio",
    "MeanVariancePortfolioState",
    "VariancePortfolio",
    "VariancePortfolioState",
]
