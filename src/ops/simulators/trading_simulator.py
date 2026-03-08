"""Trading simulator.

Provides :class:`TradingSimulator`, an :class:`Operator` that takes a
price series and a position series, tracks cash and holdings, and outputs
the total market value (scalar float64) at each timestamp.  Supports
proportional commission with an optional per-asset minimum charge.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class TradingSimulator(Operator[tuple[Series, Series], tuple[()], np.float64, dict]):
    """Simulates portfolio trading with optional commission.

    Tracks cash, positions and computes the total market value
    (cash + holdings) at each timestamp.

    Parameters
    ----------
    prices
        Vector series of asset prices.
    positions
        Vector series of desired position sizes.
    commission_rate
        Proportional commission rate applied to the absolute trade value
        of each asset.
    min_charge
        Minimum commission charged per asset when a trade occurs.
    initial_cash
        Starting cash balance.
    """

    __slots__ = ("_commission_rate", "_min_charge", "_initial_cash")

    _commission_rate: float
    _min_charge: float
    _initial_cash: float

    def __init__(
        self,
        prices: Series,
        positions: Series,
        commission_rate: float = 0.0,
        min_charge: float = 0.0,
        initial_cash: float = 0.0,
    ) -> None:
        super().__init__((prices, positions), (), np.dtype(np.float64))
        self._commission_rate = commission_rate
        self._min_charge = min_charge
        self._initial_cash = initial_cash

    @override
    def init_state(self) -> dict[str, Any]:
        return {"cash": self._initial_cash, "prev_positions": None}

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series, Series], state: dict) -> tuple[ArrayLike | None, dict]:
        prices, positions = inputs
        if not prices or not positions:
            return None, state
        current_prices = prices.values[-1]
        current_positions = positions.values[-1]

        if state["prev_positions"] is None:
            trades = current_positions
        else:
            trades = current_positions - state["prev_positions"]

        trade_cost = float(np.sum(trades * current_prices))
        trade_values = np.abs(trades * current_prices)

        if self._commission_rate > 0 or self._min_charge > 0:
            commissions = np.maximum(self._commission_rate * trade_values, self._min_charge)
            commissions = np.where(np.abs(trades) > 0, commissions, 0.0)
            total_commission = float(np.sum(commissions))
        else:
            total_commission = 0.0

        state["cash"] -= trade_cost + total_commission
        state["prev_positions"] = current_positions.copy()

        market_value = state["cash"] + float(np.sum(current_positions * current_prices))
        return market_value, state
