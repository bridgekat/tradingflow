"""Trading simulators.

Classes
-------
TradingSimulator – Tracks cash; positions and computes total market
                   value (cash + holdings) with optional proportional
                   commission and minimum charge.
"""

from .trading_simulator import TradingSimulator


__all__ = [
    "TradingSimulator",
]
