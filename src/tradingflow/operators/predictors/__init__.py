"""Prediction models with rolling retraining support.

Classes
-------
RollingPredictor              – Abstract base class with periodic
                                [`_fit`][tradingflow.operators.predictors.RollingPredictor._fit] / [`_predict`][tradingflow.operators.predictors.RollingPredictor._predict] hooks.
RollingLinearRegression       – OLS linear regression retrained on a rolling
                                window of features and scalar targets.
CrossSectionalRegression      – Cross-sectional OLS regression that fits
                                across stocks using realized forward returns.
"""

from .cross_sectional_regression import CrossSectionalRegression
from .rolling_linear_regression import RollingLinearRegression
from .rolling_predictor import RollingPredictor


__all__ = [
    "CrossSectionalRegression",
    "RollingLinearRegression",
    "RollingPredictor",
]
