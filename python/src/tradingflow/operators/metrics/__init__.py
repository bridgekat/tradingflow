r"""Financial performance and predictor-evaluation metrics.

Metric operators summarize the output of a strategy — portfolio value,
realized returns, predictions — into the familiar scalar statistics
that show up in tear sheets.  They are typically driven by a clock
input so that they update on a regular cadence (e.g. daily snapshots)
rather than on every tick.

## Portfolio-value metrics

These take a scalar `Array` input — usually the portfolio NAV emitted
by a trader — and a clock input, and produce a scalar `Array` output
containing the since-inception statistic at the most recent tick.

- [`CompoundReturn`][tradingflow.operators.metrics.compound_return.CompoundReturn] —
  annualized geometric mean return: \((P_t / P_0)^{1/n} - 1\).
- [`AverageReturn`][tradingflow.operators.metrics.average_return.AverageReturn] —
  arithmetic mean of per-period returns.
- [`Volatility`][tradingflow.operators.metrics.volatility.Volatility] —
  population standard deviation of per-period returns.
- [`SharpeRatio`][tradingflow.operators.metrics.sharpe_ratio.SharpeRatio] —
  return / risk: mean \(\div\) standard deviation of per-period returns.
- [`Drawdown`][tradingflow.operators.metrics.drawdown.Drawdown] — current
  drop from the running maximum:
  \((P_t - M_t) / M_t\) where \(M_t = \max_{s \le t} P_s\).

## Trading-activity metrics

- [`Turnover`][tradingflow.operators.metrics.turnover.Turnover] — L1 change
  in portfolio weights per rebalance, \(\|w_t - w_{t-1}\|_1\).  A proxy
  for trading cost and implementation difficulty.

## Predictor-evaluation metrics

For research workflows that compare prediction models, organized by
the kind of prediction under evaluation:

- [`mean`][tradingflow.operators.metrics.mean] — evaluators for
  mean-return predictions.  Contains the Information Coefficient
  family (Pearson or Spearman correlation between predicted and
  realized returns).
- [`variance`][tradingflow.operators.metrics.variance] — evaluators
  for covariance-matrix predictions.  Contains the realized Global
  Minimum Variance variance (how small-variance is the GMV portfolio
  built from this estimator?) and the Gaussian negative log-
  likelihood.
"""

from . import mean, variance
from .average_return import AverageReturn
from .compound_return import CompoundReturn
from .drawdown import Drawdown
from .sharpe_ratio import SharpeRatio
from .turnover import Turnover
from .volatility import Volatility

__all__ = [
    "mean",
    "variance",
    "AverageReturn",
    "CompoundReturn",
    "Drawdown",
    "SharpeRatio",
    "Turnover",
    "Volatility",
]
