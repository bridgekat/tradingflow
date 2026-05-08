"""Mean-return prediction evaluators.

- [`InformationCoefficient`][tradingflow.operators.metrics.mean.information_coefficient.InformationCoefficient]
  -cross-sectional IC / RankIC between predicted scores and
  realized forward returns.
- [`RegressionCoefficients`][tradingflow.operators.metrics.mean.regression_coefficients.RegressionCoefficients]
  - pooled OLS coefficients of a target scalar series on a baseline
  vector series, refit on each clock tick.  An intercept column is
  appended internally; the emitted vector has shape `(F + 1,)` with
  the intercept in the last position (matching `LinearRegression`).
"""

from .information_coefficient import InformationCoefficient, InformationCoefficientState
from .regression_coefficients import RegressionCoefficients, RegressionCoefficientsState

__all__ = [
    "InformationCoefficient",
    "InformationCoefficientState",
    "RegressionCoefficients",
    "RegressionCoefficientsState",
]
