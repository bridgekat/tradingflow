"""Concrete mean-predictor implementations.

- [`Sample`][tradingflow.operators.predictors.mean.sample.Sample]
  -sample mean of historical returns (baseline).
- [`SingleFeature`][tradingflow.operators.predictors.mean.single_feature.SingleFeature]
  -pass-through: returns one feature column unchanged.
- [`LinearRegression`][tradingflow.operators.predictors.mean.linear_regression.LinearRegression]
  -pooled OLS via QR decomposition.
- [`Ridge`][tradingflow.operators.predictors.mean.ridge.Ridge]
  -pooled L2-penalized regression (closed-form normal equations on
  internally-standardized features).
- [`Lasso`][tradingflow.operators.predictors.mean.lasso.Lasso]
  -pooled L1-penalized regression (CVXPY/SCS on internally-standardized
  features).

`Ridge` and `Lasso` standardize their pooled training design matrix
internally (per-feature mean/SD over `T*N` samples) and stash the
scaler with the fitted coefficients, so the L1/L2 penalty operates on
dimensionally-comparable coefficients without the caller having to
preprocess features.  This is the canonical convention used by glmnet,
sklearn, and statsmodels.  Cross-sectional
[`Standardize`][tradingflow.operators.num.standardize.Standardize] is
complementary - it reshapes each row's distribution but does not give a
time-stable per-feature scale, which is what L1/L2 regularization needs.
"""

from .sample import Sample
from .single_feature import SingleFeature
from .linear_regression import LinearRegression
from .ridge import Ridge
from .lasso import Lasso

__all__ = ["Sample", "SingleFeature", "LinearRegression", "Ridge", "Lasso"]
