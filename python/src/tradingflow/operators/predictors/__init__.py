"""Cross-sectional return and covariance predictors.

Predictors look at accumulated feature and price history, periodically
refit a statistical model, and emit a prediction for every stock in
the universe.  They are typically wrapped in a
[`Clocked`][tradingflow.operators.clocked.Clocked] operator (or paired
with a clock input) so that refitting happens on a weekly or monthly
cadence rather than on every tick.

The two abstract bases below separate the two kinds of prediction:

- [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor] —
  predicts a *return* per stock.  Fit once over a pooled sample of
  historical feature/label pairs; emit a cross-sectional vector of
  expected returns.  Feeds naturally into
  [`portfolios.mean`][tradingflow.operators.portfolios.mean] or
  [`portfolios.mean_variance`][tradingflow.operators.portfolios.mean_variance].
- [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor] —
  predicts a *covariance matrix* over the universe.  Fit on a panel
  of cross-sectional return vectors; emit an N × N matrix.  Feeds
  naturally into
  [`portfolios.variance`][tradingflow.operators.portfolios.variance] or
  [`portfolios.mean_variance`][tradingflow.operators.portfolios.mean_variance].

## Sub-modules

- [`mean`][tradingflow.operators.predictors.mean] — concrete return
  predictors (linear regression, single-feature ranking, historical
  sample mean).
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
