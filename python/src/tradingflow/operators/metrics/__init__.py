"""Performance metrics triggered by a signal series.

Both metrics take a market-value series and a signal series.  Outputs
are scalar (shape `()`) float64 values.

Classes
-------
AverageReturn  – Running cumulative mean of per-period returns.
SharpeRatio    – Annualised Sharpe ratio of per-period returns.
"""

from .average_return import AverageReturn
from .sharpe_ratio import SharpeRatio


__all__ = [
    "AverageReturn",
    "SharpeRatio",
]
