"""Trading simulators.

Classes
-------
TradingSimulator – Tracks cash, positions and computes total market
                   value (cash + holdings) with optional proportional
                   commission, minimum charge, weight-based position
                   sizing, and lot-size rounding.
"""

from .trading_simulator import TradingSimulator


__all__ = [
    "TradingSimulator",
]
