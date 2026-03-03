"""Prediction models with rolling retraining support.

Classes
-------
RollingPredictor         – Abstract base class with periodic
                           :meth:`_fit` / :meth:`_predict` hooks.
RollingLinearRegression  – OLS linear regression retrained on a rolling
                           window of features and scalar targets.
"""

from .rolling_linear_regression import RollingLinearRegression
from .rolling_predictor import RollingPredictor


__all__ = [
    "RollingPredictor",
    "RollingLinearRegression",
]
