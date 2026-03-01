"""Prediction models with rolling retraining support.

This module is the public entry point for predictors and re-exports the
predictor base class and concrete implementations from dedicated files.
"""

from .rolling_linear_regression import RollingLinearRegression
from .rolling_predictor import RollingPredictor


__all__ = [
    "RollingPredictor",
    "RollingLinearRegression",
]
