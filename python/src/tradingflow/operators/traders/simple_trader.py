"""Simple trading simulation operator."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class SimpleTraderState:
    # Configuration.
    num_stocks: int
    lot_size: float
    fee_base: float
    fee_rate: float
    trade_fn: Callable[["SimpleTraderState", np.ndarray], np.ndarray]
    verbose: bool

    # Portfolio state.
    cash: np.ndarray = field(default_factory=lambda: np.array(0.0))
    shares: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_adjust: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_close: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Rebalance signal observed on the previous tick, awaiting
    # execution at today's open.  `None` when there is no pending
    # signal.  Deferring execution by one tick keeps the signal's
    # information set strictly older than the execution price.
    pending_positions: np.ndarray | None = None

    # Transient values set by compute() before calling trade_fn().
    _current_value: np.ndarray = field(default_factory=lambda: np.array(0.0))
    _exec_price: np.ndarray = field(default_factory=lambda: np.empty(0))


class OHLCV(IntEnum):
    """Column indices within the prices array of shape `(num_stocks, 5)`."""

    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4


class SimpleTrader(
    Operator[
        ArrayView[np.float64],  # soft positions (num_stocks,)
        ArrayView[np.float64],  # OHLCV prices (num_stocks, 5)
        ArrayView[np.float64],  # adjusts (num_stocks,)
        ArrayView[np.float64],  # output: (position_value, excess_liquidity)
        SimpleTraderState,
    ]
):
    """Simple trading simulation operator.

    On every tick:

    1. Adjusts held shares for dividend reinvestment via forward
       adjustment factor changes.
    2. If a rebalance signal was received on the **previous** tick,
       first force-liquidates held positions in stocks with no valid
       exec price today (suspended or delisted) at their last valid
       close - each forced sale charged the same fee as a normal
       trade - then calls `trade_fn` with the post-liquidation state
       to obtain net-delta lots for the rest, and executes those lots
       at today's open.  Tradable stocks therefore incur at most one
       trade (one fee) per rebalance, not two.  Suspended / delisted
       stocks receive zero shares post-rebalance.
    3. If the soft-positions input was updated this tick, stores it as
       the pending signal for execution on the **next** tick.  Trading
       is therefore deferred by one tick, so today's signal (derived
       from data observable at today's close) is executed at tomorrow's
       open - the upstream signal construction never has access to the
       execution-price information set.
    4. Outputs a 2-element array `(holdings_value, cash)` where
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

    Notes
    -----
    The rebalance cadence is controlled by upstream: the trader
    rebalances exactly when the soft-positions input produces (which in
    turn is driven by the predictor's clock trigger).

    This operator uses a simplified market model:

    - **Fixed market impact**: transaction costs are modelled as a flat
        fee plus a proportional rate.  There is no slippage, spread, or
        volume-dependent impact.
    - **One-tick-delayed execution**: a signal received this tick is
        executed at the opening price of the **next** tick.

    These assumptions are a reasonable approximation when trading sizes
    are small relative to market liquidity, but become inaccurate as
    capital grows large or stocks are illiquid.

    **NaN prices (suspended / delisted stocks).**  A stock with `NaN`
    open or close on the current tick is handled as follows:

    - **Valuation (every tick)**: portfolio value uses the *last valid
      close* carried forward, so held shares of a suspended stock keep
      contributing at their most recent known price instead of dropping
      out.
    - **Rebalance**: stocks with a finite, positive current open are
      rebalanced via a single net trade (delta lots from `trade_fn`),
      paying one round of fees.  Stocks with non-finite or non-positive
      open cannot be traded on the open market; if we held any, the
      simulator force-closes them at their last valid close (charged the
      same fee as a normal trade - an idealisation that is not
      achievable in live trading) so they hold zero shares
      post-rebalance.  No fresh entry is ever made into a stock whose
      current open is invalid - `trade_fn` is free to return lots for
      such stocks, but the per-stock execution loop enforces the skip.
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

        super().__init__(
            inputs=(soft_positions, prices, adjusts),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(2,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> SimpleTraderState:
        n = self._num_stocks
        return SimpleTraderState(
            num_stocks=n,
            lot_size=self._lot_size,
            fee_base=self._fee_base,
            fee_rate=self._fee_rate,
            trade_fn=self._trade_fn,
            verbose=self._verbose,
            cash=np.array(self._initial_cash),
            shares=np.zeros(n),
            last_adjust=np.ones(n),
            last_close=np.full(n, np.nan),
        )

    @staticmethod
    def compute(
        state: SimpleTraderState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
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
        valid_adjusts = np.isfinite(adjusts) & (adjusts > 0)
        adjust_mask = valid_adjusts & (state.last_adjust > 0)
        state.shares[adjust_mask] *= adjusts[adjust_mask] / state.last_adjust[adjust_mask]
        state.last_adjust[valid_adjusts] = adjusts[valid_adjusts]

        # Execute pending rebalance from the previous tick: signal
        # observed one tick ago, executed at today's open, sized at
        # yesterday's close.  Performed BEFORE the `last_close` update
        # so valuation uses yesterday's close (= the close of the tick
        # on which the signal was generated) - this eliminates both the
        # intraday open/close lookahead and the upstream "signal sees
        # today's data, trades at today's open" lookahead in one step.
        traded = False
        if state.pending_positions is not None:
            pending = state.pending_positions
            valid_exec = np.isfinite(opens) & (opens > 0)

            # Step 1: force-liquidate held positions in stocks with no
            # valid exec price today (suspended or delisted) at their
            # last valid close - the simulator assumes an idealised
            # exit even when no open-market trade is actually possible.
            # Each forced sale is charged the same fee as a normal
            # trade.
            force_liq = (state.shares != 0) & ~valid_exec & np.isfinite(state.last_close)
            sell_values = state.shares[force_liq] * state.last_close[force_liq]
            fees = np.maximum(state.fee_base, np.abs(sell_values) * state.fee_rate)
            state.cash += np.sum(sell_values - fees)
            state.shares[force_liq] = 0.0
            traded |= bool(force_liq.any())

            # Step 2: compute portfolio value post force-liquidation,
            # then ask `trade_fn` for net delta lots.  Tradable stocks
            # therefore incur at most one trade (and one fee) per
            # rebalance, not two.
            held = (state.shares != 0) & np.isfinite(state.last_close)
            state._current_value = state.cash + np.sum(state.shares[held] * state.last_close[held])
            state._exec_price = opens
            trade_lots = state.trade_fn(state, pending)

            # Step 3: execute the net delta lots at today's open for
            # tradable stocks only.  Per-stock loop because of
            # sub-lot-remnant liquidation and per-trade fees.
            for i in range(N):
                if not valid_exec[i]:
                    continue
                p = opens[i]

                # Get share counts from lot counts.
                trade_shares = trade_lots[i] * state.lot_size

                # Liquidate sub-lot remnants.
                if abs(state.shares[i] + trade_shares) < state.lot_size:
                    trade_shares = -state.shares[i]

                if trade_shares != 0:
                    trade_value = trade_shares * p
                    fee = max(state.fee_base, abs(trade_value) * state.fee_rate)
                    state.cash -= trade_value + fee
                    state.shares[i] += trade_shares
                    traded = True

            if traded and state.verbose:
                held_mask = np.abs(state.shares) >= state.lot_size
                if held_mask.any():
                    idx = np.where(held_mask)[0]
                    print(f"  positions: { {int(i): state.shares[i] for i in idx} }")

            state.pending_positions = None

        # Store new rebalance signal for execution on the NEXT tick.
        # `.value()` already returns a fresh copy, but `.copy()` makes
        # the one-tick-delay intent explicit at the storage site.
        if produced[0]:
            state.pending_positions = soft_positions.copy()

        # Update the last-valid-close carry-forward for stocks that
        # ticked this cycle; suspended stocks retain their previous
        # last-valid close.  Done AFTER the rebalance so today's close
        # is only ever used for end-of-day mark-to-market.
        close_tick = np.isfinite(closes)
        state.last_close[close_tick] = closes[close_tick]

        # Output `(holdings_value, cash)`.  Total portfolio value = sum.
        held = (state.shares != 0) & np.isfinite(state.last_close)
        holdings_value = np.sum(state.shares[held] * state.last_close[held])
        output.write(np.array([holdings_value, state.cash], dtype=np.float64))
        return True
