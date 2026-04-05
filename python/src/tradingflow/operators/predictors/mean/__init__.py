"""Concrete mean-predictor implementations.

- [`LinearRegression`][tradingflow.operators.predictors.mean.LinearRegression]
  -- pooled OLS via QR decomposition.
"""

from .linear_regression import LinearRegression

__all__ = ["LinearRegression"]
