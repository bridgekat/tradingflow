"""Concrete mean-predictor implementations.

- [`Sample`][tradingflow.operators.predictors.mean.sample.Sample]
  —sample mean of historical returns (baseline).
- [`SingleFeature`][tradingflow.operators.predictors.mean.single_feature.SingleFeature]
  —pass-through: returns one feature column unchanged.
- [`LinearRegression`][tradingflow.operators.predictors.mean.linear_regression.LinearRegression]
  —pooled OLS via QR decomposition.
"""

from .sample import Sample
from .single_feature import SingleFeature
from .linear_regression import LinearRegression

__all__ = ["Sample", "SingleFeature", "LinearRegression"]
