"""Trading simulator operator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ... import Operator, Series


class TradingSimulator(Operator[dict[str, Any], np.float64]):
    """Simulates portfolio trading with configurable commissions."""

    __slots__ = ("_commission_rate", "_min_charge")

    def __init__(
        self,
        prices: Series[Any],
        positions: Series[Any],
        commission_rate: float = 0.0,
        min_charge: float = 0.0,
        initial_cash: float = 0.0,
    ) -> None:
        if len(prices.shape) != 1 or prices.shape != positions.shape:
            raise ValueError(
                "prices and positions must be 1-D series with the same "
                f"shape; got prices.shape={prices.shape}, "
                f"positions.shape={positions.shape}"
            )
        state: dict[str, Any] = {
            "prev_positions": np.zeros(prices.shape[0], dtype=np.float64),
            "cash": float(initial_cash),
        }
        super().__init__([prices, positions], state, np.dtype(np.float64))
        self._commission_rate = commission_rate
        self._min_charge = min_charge

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        price_series, position_series = inputs[0], inputs[1]
        if not price_series or not position_series:
            return None

        prices: NDArray[np.float64] = np.asarray(price_series.values[-1], dtype=np.float64)
        new_positions: NDArray[np.float64] = np.asarray(position_series.values[-1], dtype=np.float64)
        prev_positions: NDArray[np.float64] = self._state["prev_positions"]

        trades = new_positions - prev_positions
        trade_values = np.abs(trades * prices)

        traded_mask = trades != 0.0
        commissions = np.where(
            traded_mask,
            np.maximum(self._commission_rate * trade_values, self._min_charge),
            0.0,
        )
        total_commission = float(commissions.sum())

        trade_cost = float((trades * prices).sum())

        self._state["cash"] -= trade_cost + total_commission
        self._state["prev_positions"] = new_positions.copy()

        market_value = self._state["cash"] + float((new_positions * prices).sum())
        return market_value
