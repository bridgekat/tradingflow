"""Cross-sectional return and covariance predictors.

Predictor operators read accumulated feature and price history from
upstream `Series` inputs (produced by `Record` operators),
periodically fit a model on historical data, and output predictions
for every stock.

All operators in this module are [`Operator`][tradingflow.Operator]
subclasses whose [`compute`][tradingflow.Operator.compute] method runs in
Python.

- [`MeanPredictor`][tradingflow.operators.predictors.MeanPredictor] --
  abstract base for return prediction (pools individual samples).
- [`VariancePredictor`][tradingflow.operators.predictors.VariancePredictor] --
  abstract base for covariance prediction (cross-sectional return vectors).

## Sub-modules

- [`mean`][tradingflow.operators.predictors.mean] -- concrete mean-predictor
  implementations.
- [`variance`][tradingflow.operators.predictors.variance] -- concrete
  variance-predictor implementations.
"""

from . import mean
from . import variance
from .mean_predictor import MeanPredictor, MeanPredictorState
from .variance_predictor import VariancePredictor, VariancePredictorState

__all__ = [
    "mean",
    "variance",
    "MeanPredictor",
    "MeanPredictorState",
    "VariancePredictor",
    "VariancePredictorState",
]
