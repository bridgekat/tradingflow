"""Trading simulators.

This module is the public entry point for simulator operators and re-exports
implementations from dedicated files.
"""

from .trading_simulator import TradingSimulator


__all__ = [
    "TradingSimulator",
]
