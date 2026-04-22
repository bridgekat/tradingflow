"""Cross-sectional predictors of a target and its covariance.

Predictors look at accumulated feature and target history, periodically
refit a statistical model, and emit a prediction for every stock in
the universe.  They are typically driven by a rebalance-clocked
`universe` input so refitting happens on a weekly or monthly cadence
rather than on every tick.

The two abstract bases below separate the two kinds of prediction:

- [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor] —
  predicts the target value per stock.  Fit once over a pooled sample
  of aligned feature/target rows; emit a cross-sectional vector.  Feeds
  naturally into
  [`portfolios.mean`][tradingflow.operators.portfolios.mean] or
  [`portfolios.mean_variance`][tradingflow.operators.portfolios.mean_variance].
- [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor] —
  predicts the target's covariance matrix over the universe.  Emit
  an N × N matrix.  Feeds naturally into
  [`portfolios.variance`][tradingflow.operators.portfolios.variance] or
  [`portfolios.mean_variance`][tradingflow.operators.portfolios.mean_variance].

## Prediction contract

Both bases take a `target_series` input (rather than computing returns
internally from prices).  The **meaning of the emitted prediction is
defined by how this target series is constructed upstream** in the
graph.  Common choices:

- `Record(PctChange(prices))` — linear returns.  Predictors emit
  predicted linear returns / covariances, suitable for feeding into
  `MeanVariancePortfolio` whose objective is in linear-return units.
- `Record(Diff(Log(prices)))` — log returns.  Predictions are then in
  log-return units.
- `Record(Gaussianize(PctChange(prices)))` — rank-transformed returns.
  Predictions are on the rank scale; suitable for
  `MeanPortfolio` subclasses whose rebalance logic only consumes
  *ordering* (top-N, rank-linear, softmax).  **Not suitable** for
  `MeanVariancePortfolio` / `VariancePortfolio`, which require
  magnitudes.

The `target_delay` parameter expresses how many periods the target
series lags the features series (e.g. an h-period forward return is
only observable h ticks late).  At every recompute the invariant
`len(features_series) == len(target_series) + target_delay` is
asserted; misalignment raises `AssertionError` at runtime.

## Sub-modules

- [`mean`][tradingflow.operators.predictors.mean] — concrete mean
  predictors (historical sample mean, single-feature pass-through,
  pooled OLS linear regression).
- [`variance`][tradingflow.operators.predictors.variance] — concrete
  covariance estimators (sample covariance, Ledoit-Wolf / Schafer-
  Strimmer shrinkage, single-index, hierarchical, random-matrix
  theory).
"""

from . import mean, variance
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
