"""Financial metrics operators.

Clock-driven since-inception metrics that take a scalar Array input and
produce a scalar Array output:

- [`CompoundReturn`][tradingflow.operators.metrics.CompoundReturn]
  -- `(current / first)^(1/n) - 1`
- [`AverageReturn`][tradingflow.operators.metrics.AverageReturn]
  -- mean of period returns
- [`Volatility`][tradingflow.operators.metrics.Volatility]
  -- population standard deviation of period returns
- [`SharpeRatio`][tradingflow.operators.metrics.SharpeRatio]
  -- mean / std of period returns
- [`Drawdown`][tradingflow.operators.metrics.Drawdown]
  -- `(current - running_max) / running_max`
- [`Turnover`][tradingflow.operators.metrics.Turnover]
  -- `||w_t - w_{t-1}||_1`: L1 change in portfolio weights per rebalance

Predictor evaluation metrics live in submodules organised by the kind
of prediction they evaluate:

- [`mean`][tradingflow.operators.metrics.mean] -- evaluators for
  mean-return predictions (e.g. Information Coefficient).
- [`variance`][tradingflow.operators.metrics.variance] -- evaluators
  for covariance-matrix predictions (e.g. realized GMV variance,
  Gaussian negative log-likelihood).
"""

from . import mean
from . import variance
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
