"""Rolling statistical filters and technical indicators.

All filters extend :class:`Rolling` (or :class:`Operator` directly when
the output shape differs from the input) and accept either a count-based
(``int``) or time-based (``np.timedelta64``) window.

Classes
-------
Rolling           – Abstract base providing :meth:`_get_window`.
MovingAverage     – Simple moving average (SMA).
WeightedMovingAverage – Linearly weighted moving average (WMA, int window only).
ExponentialMovingAverage – Exponential moving average (EMA).
MovingVariance    – Rolling sample variance.
MovingCovariance  – Rolling sample covariance matrix (vector input → matrix output).
Momentum          – Difference between latest and lagged value.
BollingerBand     – Rolling mean ± k times rolling standard deviation.
"""

from .bollinger_band import BollingerBand
from .exponential_moving_average import ExponentialMovingAverage
from .momentum import Momentum
from .moving_average import MovingAverage
from .moving_covariance import MovingCovariance
from .moving_variance import MovingVariance
from .rolling import Rolling
from .weighted_moving_average import WeightedMovingAverage


__all__ = [
    "BollingerBand",
    "ExponentialMovingAverage",
    "Momentum",
    "MovingAverage",
    "MovingCovariance",
    "MovingVariance",
    "Rolling",
    "WeightedMovingAverage",
]
