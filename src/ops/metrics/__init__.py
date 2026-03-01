"""Performance metrics that update on a periodic signal.

This module is the public entry point for metric operators and re-exports
implementations from dedicated files.
"""

from .average_return import AverageReturn
from .sharpe_ratio import SharpeRatio


__all__ = [
    "AverageReturn",
    "SharpeRatio",
]
