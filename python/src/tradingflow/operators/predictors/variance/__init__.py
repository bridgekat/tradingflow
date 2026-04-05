"""Concrete variance-predictor implementations.

- [`Sample`][tradingflow.operators.predictors.variance.Sample]
  -- sample covariance of historical returns (baseline).
- [`Shrinkage`][tradingflow.operators.predictors.variance.Shrinkage]
  -- Ledoit-Wolf linear shrinkage estimator.
"""

from .sample import Sample
from .shrinkage import Shrinkage

__all__ = ["Sample", "Shrinkage"]
