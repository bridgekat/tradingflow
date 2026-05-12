"""Concrete mean-variance portfolio implementations.

- [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]
  -Markowitz mean-variance optimization via CVXPY, with a pluggable
  [`Mode`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode] selecting
  among four equivalent formulations (min-variance-given-return,
  max-return-given-variance, variance-penalized, std-dev-penalized).
- [`MarkowitzADMM`][tradingflow.operators.portfolios.mean_variance.markowitz_admm.MarkowitzADMM]
  -Markowitz mean-variance optimization via augmented-saddle ADMM
  (NumPy/SciPy only, no CVXPY at runtime; robust to rank-deficient
  equality constraints).  Restricted to the
  [`MIN_MEAN_VARIANCE`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
  mode.
"""

from .markowitz import Markowitz, Mode
from .markowitz_admm import MarkowitzADMM

__all__ = [
    "Markowitz",
    "MarkowitzADMM",
    "Mode",
]
