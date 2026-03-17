"""Trading simulator.

Provides [`TradingSimulator`][tradingflow.operators.simulators.TradingSimulator], an [`Operator`][tradingflow.Operator] that takes a
price series and a position/weight series, tracks cash and holdings, and
outputs the total market value (scalar float64) at each timestamp.  Supports
proportional commission with an optional per-asset minimum charge, lot-size
rounding, and weight-based position sizing.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator
from ...observable import Observable


class TradingSimulator(Operator[tuple[Observable, Observable], tuple[()], np.float64, dict]):
    """Simulates portfolio trading with optional commission.

    Tracks cash, positions and computes the total market value
    (cash + holdings) at each timestamp.

    Parameters
    ----------
    prices
        Vector series of asset prices.
    positions
        Vector series of desired position sizes, or portfolio weights
        when `weight_mode=True`.
    commission_rate
        Proportional commission rate applied to the absolute trade value
        of each asset.
    min_charge
        Minimum commission charged per asset when a trade occurs.
    initial_cash
        Starting cash balance.
    weight_mode
        When `True`, the *positions* input is interpreted as portfolio
        weights (summing to ~1).  Target positions are computed as
        `floor(market_value * weight / (price * lot_size)) * lot_size`.
    lot_size
        Minimum tradeable unit; positions are rounded down to multiples
        of this value in weight mode.  Default `1`.
    """

    __slots__ = ("_commission_rate", "_min_charge", "_initial_cash", "_weight_mode", "_lot_size")

    _commission_rate: float
    _min_charge: float
    _initial_cash: float
    _weight_mode: bool
    _lot_size: int

    def __init__(
        self,
        prices: Observable,
        positions: Observable,
        commission_rate: float = 0.0,
        min_charge: float = 0.0,
        initial_cash: float = 0.0,
        weight_mode: bool = False,
        lot_size: int = 1,
    ) -> None:
        super().__init__((prices, positions), (), np.dtype(np.float64))
        self._commission_rate = commission_rate
        self._min_charge = min_charge
        self._initial_cash = initial_cash
        self._weight_mode = weight_mode
        self._lot_size = lot_size

    @override
    def init_state(self) -> dict[str, Any]:
        return {"cash": self._initial_cash, "prev_positions": None}

    @override
    def compute(
        self, timestamp: np.datetime64, inputs: tuple[Observable, Observable], state: dict
    ) -> tuple[ArrayLike | None, dict]:
        prices, positions = inputs
        if not prices or not positions:
            return None, state
        current_prices = prices.last
        current_positions = positions.last

        if self._weight_mode:
            # Convert weights to positions using current portfolio value
            if state["prev_positions"] is None:
                market_value = self._initial_cash
            else:
                market_value = state["cash"] + float(np.nansum(state["prev_positions"] * current_prices))
            weights = current_positions
            safe_prices = np.where(current_prices > 0, current_prices, np.inf)
            raw_positions = market_value * weights / safe_prices
            lot = self._lot_size
            current_positions = (np.floor(raw_positions / lot) * lot).astype(np.float64)

        if state["prev_positions"] is None:
            trades = current_positions
        else:
            trades = current_positions - state["prev_positions"]

        trade_cost = float(np.nansum(trades * current_prices))
        trade_values = np.abs(trades * current_prices)

        if self._commission_rate > 0 or self._min_charge > 0:
            commissions = np.maximum(self._commission_rate * trade_values, self._min_charge)
            commissions = np.where(np.abs(trades) > 0, commissions, 0.0)
            total_commission = float(np.nansum(commissions))
        else:
            total_commission = 0.0

        state["cash"] -= trade_cost + total_commission
        state["prev_positions"] = current_positions.copy()

        market_value = state["cash"] + float(np.nansum(current_positions * current_prices))
        return market_value, state
