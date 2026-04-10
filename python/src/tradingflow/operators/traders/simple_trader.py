"""Simple trading simulation operator."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ...views import ArrayView
from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind
from ...utils import coerce_timestamp


@dataclass(slots=True)
class SimpleTraderState:
    # Configuration.
    num_stocks: int
    lot_size: float
    fee_base: float
    fee_rate: float
    trade_fn: Callable[["SimpleTraderState", np.ndarray], np.ndarray]
    verbose: bool
    trading_start: int | None

    # Portfolio state.
    cash: float = 0.0
    shares: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_adjust: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Transient values set by compute() before calling trade_fn().
    _current_value: float = 0.0
    _exec_price: np.ndarray = field(default_factory=lambda: np.empty(0))


class OHLCV(IntEnum):
    """Column indices within the prices array of shape ``(num_stocks, 5)``."""

    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4


class SimpleTrader(
    Operator[
        tuple[
            Handle[Array[np.float64]],  # soft positions (num_stocks,)
            Handle[Array[np.float64]],  # OHLCV prices (num_stocks, 5)
            Handle[Array[np.float64]],  # adjusts (num_stocks,)
        ],
        Handle[Array[np.float64]],  # (position_value, excess_liquidity)
        SimpleTraderState,
    ]
):
    """Simple trading simulation operator.

    On every tick:

    1. Adjusts held shares for dividend reinvestment via forward
       adjustment factor changes.
    2. If the soft-positions input was updated (rebalance signal),
       computes the current portfolio value, calls `trade_fn` to get
       the number of lots to trade per stock, executes the trades at
       opening prices, and deducts transaction fees.  If the resulting
       absolute position for a stock is less than one lot, the remainder
       is liquidated.
    3. Outputs a 2-element array ``(holdings_value, cash)`` where
       *holdings_value* is positions valued at closing prices and
       *cash* is the cash balance.  Total portfolio value is their sum.

    Parameters
    ----------
    soft_positions
        Soft position weights, shape `(num_stocks,)`.
    prices
        Stacked OHLCV prices, shape `(num_stocks, 5)`.  Columns are
        `(open, high, low, close, volume)`.
    adjusts
        Stacked forward adjustment factors, shape `(num_stocks,)`.
    trade_fn
        `(state, soft_positions) -> lots`.  Called with the current
        state and the soft position-weight vector of shape
        `(num_stocks,)`.  Must return an array of shape
        `(num_stocks,)` representing the number of lots to trade
        (positive for buy, negative for sell).
    initial_cash
        Starting capital.
    lot_size
        Minimum trade lot size (number of shares).
    fee_base
        Minimum transaction fee per trade.
    fee_rate
        Proportional transaction fee rate.
    verbose
        If `True`, print current positions after each rebalance.
    trading_start
        If set, rebalance signals before this timestamp are ignored.

    Notes
    -----

    This operator uses a simplified market model:

    - **Fixed market impact**: transaction costs are modelled as a flat
        fee plus a proportional rate.  There is no slippage, spread, or
        volume-dependent impact.
    - **Immediate execution**: all trades execute instantly at the
        opening price of the current tick.

    These assumptions are a reasonable approximation when trading sizes
    are small relative to market liquidity, but become inaccurate as
    capital grows large or stocks are illiquid.
    """

    def __init__(
        self,
        soft_positions: Handle,
        prices: Handle,
        adjusts: Handle,
        *,
        trade_fn: Callable[[SimpleTraderState, np.ndarray], np.ndarray],
        initial_cash: float,
        lot_size: float,
        fee_base: float,
        fee_rate: float,
        verbose: bool = False,
        trading_start: np.datetime64 | None = None,
    ) -> None:
        assert len(soft_positions.shape) == 1, "Soft positions input must have shape (num_stocks,)."
        assert (
            len(prices.shape) == 2 and prices.shape[1] == 5
        ), "Prices input must have shape (num_stocks, 5) where the 5 columns are (open, high, low, close, volume)."
        assert (
            soft_positions.shape[0] == prices.shape[0]
        ), "Soft positions and prices must have the same number of stocks."

        self._num_stocks = soft_positions.shape[0]
        self._trade_fn = trade_fn
        self._initial_cash = initial_cash
        self._lot_size = lot_size
        self._fee_base = fee_base
        self._fee_rate = fee_rate
        self._verbose = verbose
        self._trading_start = int(coerce_timestamp(trading_start)) if trading_start is not None else None

        super().__init__(
            inputs=(soft_positions, prices, adjusts),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(2,),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> SimpleTraderState:
        n = self._num_stocks
        return SimpleTraderState(
            num_stocks=n,
            lot_size=self._lot_size,
            fee_base=self._fee_base,
            fee_rate=self._fee_rate,
            trade_fn=self._trade_fn,
            verbose=self._verbose,
            trading_start=self._trading_start,
            cash=self._initial_cash,
            shares=np.zeros(n),
            last_adjust=np.ones(n),
        )

    @staticmethod
    def compute(
        state: SimpleTraderState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        N = state.num_stocks
        soft_positions = inputs[0].value()
        prices = inputs[1].value()
        adjusts = inputs[2].value()
        opens = prices[:, OHLCV.OPEN]
        closes = prices[:, OHLCV.CLOSE]

        # Adjust shares for dividends (reinvesting all dividends).
        valid_adjusts = np.isfinite(adjusts) & (adjusts > 0)
        valid_last_adjusts = state.last_adjust > 0
        adjust_mask = valid_adjusts & valid_last_adjusts
        state.shares[adjust_mask] *= adjusts[adjust_mask] / state.last_adjust[adjust_mask]
        state.last_adjust[valid_adjusts] = adjusts[valid_adjusts]

        # Compute portfolio market value.
        held = (state.shares != 0) & np.isfinite(closes)
        state._current_value = state.cash + np.sum(state.shares[held] * closes[held])

        # Rebalance if soft positions input was updated.
        rebalance = notify.input_produced()[0]
        if state.trading_start is not None and timestamp < state.trading_start:
            rebalance = False
        if rebalance:

            # Execution price = open price.
            state._exec_price = opens

            # Ask trade_fn for lot counts.
            trade_lots = state.trade_fn(state, soft_positions)

            traded = False
            for i in range(N):
                p = state._exec_price[i]
                if not np.isfinite(p) or p <= 0:
                    continue

                # Get share counts from lot counts.
                trade_shares = trade_lots[i] * state.lot_size

                # Liquidate sub-lot remnants.
                if abs(state.shares[i] + trade_shares) < state.lot_size:
                    trade_shares = -state.shares[i]

                # Simulate trade.
                if trade_shares != 0:
                    trade_value = trade_shares * p
                    state.cash -= trade_value
                    fee = max(state.fee_base, abs(trade_value) * state.fee_rate)
                    state.cash -= fee
                    state.shares[i] += trade_shares
                    traded = True

            if traded and state.verbose:
                held_mask = np.abs(state.shares) >= state.lot_size
                if held_mask.any():
                    idx = np.where(held_mask)[0]
                    print(f"  positions: { {int(i): state.shares[i] for i in idx} }")

        # Output (holdings_value, cash).  Total portfolio value = sum.
        held = (state.shares != 0) & np.isfinite(closes)
        holdings_value = np.sum(state.shares[held] * closes[held])
        output.write(np.array([holdings_value, state.cash], dtype=np.float64))
        return True
