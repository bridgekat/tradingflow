"""Benchmark operator — frictionless ideal portfolio replication."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...operator import Operator
from ...types import Handle, NodeKind
from ...views import ArrayView
from ..traders.simple_trader import OHLCV


@dataclass(slots=True)
class BenchmarkState:
    # Configuration.
    num_stocks: int
    initial_cash: float
    use_adjusts: bool

    # Portfolio state.
    cash: float = 0.0
    shares: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_adjust: np.ndarray = field(default_factory=lambda: np.empty(0))


class Benchmark(
    Operator[
        ArrayView[np.float64],  # soft positions (num_stocks,)
        ArrayView[np.float64],  # OHLCV prices (num_stocks, 5)
        ArrayView[np.float64],  # adjusts (num_stocks,)
        ArrayView[np.float64],  # output: (holdings_value, cash)
        BenchmarkState,
    ]
):
    """Frictionless benchmark that replicates soft position weights exactly.

    On every tick:

    1. Adjusts held shares for dividend reinvestment via forward
       adjustment factor changes.
    2. If the soft-positions input was updated (rebalance signal),
       computes the current portfolio value and rebalances to exactly
       match the target weights: target shares = `soft_positions *
       current_value / open_price`, executed at opening prices as
       fractional shares with no transaction fees and no lot rounding.
    3. Outputs a 2-element array `(holdings_value, cash)` where
       `holdings_value` is positions valued at closing prices and
       `cash` is the cash balance.  Total portfolio value is their sum.

    Parameters
    ----------
    soft_positions
        Soft position weights, shape `(num_stocks,)`.
    prices
        Stacked OHLCV prices, shape `(num_stocks, 5)`.  Columns are
        `(open, high, low, close, volume)`.
    adjusts
        Stacked forward adjustment factors, shape `(num_stocks,)`.
    initial_cash
        Starting capital.
    use_adjusts
        If `True`, account for dividend reinvestment via adjustment
        factors (total return index).  If `False`, use raw prices
        (price index).

    Notes
    -----
    The rebalance cadence is controlled by upstream: the benchmark
    rebalances exactly when the soft-positions input produces.
    """

    def __init__(
        self,
        soft_positions: Handle,
        prices: Handle,
        adjusts: Handle,
        *,
        initial_cash: float,
        use_adjusts: bool,
    ) -> None:
        assert len(soft_positions.shape) == 1, "Soft positions input must have shape (num_stocks,)."
        assert (
            len(prices.shape) == 2 and prices.shape[1] == 5
        ), "Prices input must have shape (num_stocks, 5) where the 5 columns are (open, high, low, close, volume)."
        assert (
            soft_positions.shape[0] == prices.shape[0]
        ), "Soft positions and prices must have the same number of stocks."

        self._num_stocks = soft_positions.shape[0]
        self._initial_cash = initial_cash
        self._use_adjusts = use_adjusts

        super().__init__(
            inputs=(soft_positions, prices, adjusts),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(2,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[
            ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]
        ],
        timestamp: int,
    ) -> BenchmarkState:
        n = self._num_stocks
        return BenchmarkState(
            num_stocks=n,
            initial_cash=self._initial_cash,
            use_adjusts=self._use_adjusts,
            cash=self._initial_cash,
            shares=np.zeros(n),
            last_adjust=np.ones(n),
        )

    @staticmethod
    def compute(
        state: BenchmarkState,
        inputs: tuple[
            ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]
        ],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        N = state.num_stocks
        soft_positions = inputs[0].value()
        prices = inputs[1].value()
        adjusts = inputs[2].value()
        opens = prices[:, OHLCV.OPEN]
        closes = prices[:, OHLCV.CLOSE]

        # Adjust shares for dividends (reinvesting all dividends).
        if state.use_adjusts:
            valid_adjusts = np.isfinite(adjusts) & (adjusts > 0)
            valid_last_adjusts = state.last_adjust > 0
            adjust_mask = valid_adjusts & valid_last_adjusts
            state.shares[adjust_mask] *= adjusts[adjust_mask] / state.last_adjust[adjust_mask]
            state.last_adjust[valid_adjusts] = adjusts[valid_adjusts]

        # Compute portfolio market value.
        held = (state.shares != 0) & np.isfinite(closes)
        current_value = state.cash + np.sum(state.shares[held] * closes[held])

        # Rebalance if soft positions input was updated.
        rebalance = produced[0]
        if rebalance:

            # Execution price = open price.
            exec_price = opens

            for i in range(N):
                p = exec_price[i]
                if not np.isfinite(p) or p <= 0:
                    continue

                # Calculate trade shares.
                target_value = soft_positions[i] * current_value
                target_shares = target_value / p
                trade_shares = target_shares - state.shares[i]

                # Simulate frictionless trade.
                trade_value = trade_shares * p
                state.cash -= trade_value
                state.shares[i] += trade_shares

        # Output (holdings_value, cash).  Total portfolio value = sum.
        held = (state.shares != 0) & np.isfinite(closes)
        holdings_value = np.sum(state.shares[held] * closes[held])
        output.write(np.array([holdings_value, state.cash], dtype=np.float64))
        return True
