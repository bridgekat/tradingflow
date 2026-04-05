"""Concrete mean-predictor implementations.

- [`Sample`][tradingflow.operators.predictors.mean.Sample]
  -- sample mean of historical returns (baseline).
- [`LinearRegression`][tradingflow.operators.predictors.mean.LinearRegression]
  -- pooled OLS via QR decomposition.
"""

from .sample import Sample
from .linear_regression import LinearRegression

__all__ = ["Sample", "LinearRegression"]
