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

Predictor evaluation metrics:

- [`InformationCoefficient`][tradingflow.operators.metrics.InformationCoefficient]
  -- cross-sectional IC or RankIC between predicted scores and
  realized forward returns, for evaluating mean-return predictions
- [`MinimumVariance`][tradingflow.operators.metrics.MinimumVariance]
  -- realized variance of the global minimum variance portfolio,
  for evaluating covariance matrix predictions
"""

from .compound_return import CompoundReturn
from .average_return import AverageReturn
from .volatility import Volatility
from .sharpe_ratio import SharpeRatio
from .drawdown import Drawdown
from .information_coefficient import InformationCoefficient
from .minimum_variance import MinimumVariance

__all__ = [
    "CompoundReturn",
    "AverageReturn",
    "Volatility",
    "SharpeRatio",
    "Drawdown",
    "InformationCoefficient",
    "MinimumVariance",
]
