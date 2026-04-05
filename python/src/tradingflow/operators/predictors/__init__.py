"""Cross-sectional return predictors.

Predictor operators accumulate per-tick cross-sectional feature and price
snapshots, periodically fit a model on historical data, and output
predicted future returns for every stock.

All operators in this module are [`Operator`][tradingflow.Operator]
subclasses whose [`compute`][tradingflow.Operator.compute] method runs in
Python.

- [`MeanPredictor`][tradingflow.operators.predictors.MeanPredictor] --
  abstract base that handles ring-buffer accumulation, sample collection,
  and rebalance scheduling.

## Sub-modules

- [`mean`][tradingflow.operators.predictors.mean] -- concrete mean-predictor
  implementations.
"""

from . import mean
from .mean_predictor import MeanPredictor, MeanPredictorState

__all__ = [
    "mean",
    "MeanPredictor",
    "MeanPredictorState",
]
