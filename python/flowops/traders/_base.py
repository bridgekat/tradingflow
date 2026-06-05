"""Shared base for trader operators: SimpleTrader + its state.

Ported from ``tradingflow.operators.traders.simple_trader``.  The
``init`` / ``compute`` bodies are kept verbatim; only the framework
imports and handle-based ``__init__`` are replaced with plain config
scalars (``num_stocks`` taken as a build kwarg or read from the input
view at ``init``).
"""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field

import numpy as np


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
    # execution at today's close.  `None` when there is no pending
    # signal.  Deferring execution by one tick keeps the signal's
    # information set strictly older than the execution price.
    pending_positions: np.ndarray | None = None

    # Transient values set by compute() before calling trade_fn().
    _current_value: np.ndarray = field(default_factory=lambda: np.array(0.0))
    _exec_price: np.ndarray = field(default_factory=lambda: np.empty(0))


class SimpleTrader:
    """Simple trading simulation operator.

    Inputs (all array views, shape ``(num_stocks,)``):
        0. soft positions
        1. close prices (execution + mark-to-market)
        2. adjusts (forward adjustment factors)
        3. upper price limit
        4. lower price limit
    Output: array view of shape ``(2,)`` -> ``(holdings_value, cash)``.

    See the upstream docstring for the full market-model description.
    """

    def __init__(
        self,
        *,
        num_stocks: int,
        trade_fn: Callable[[SimpleTraderState, np.ndarray], np.ndarray],
        initial_cash: float,
        lot_size: float,
        fee_base: float,
        fee_rate: float,
        verbose: bool = False,
    ) -> None:
        self._num_stocks = num_stocks
        self._trade_fn = trade_fn
        self._initial_cash = initial_cash
        self._lot_size = lot_size
        self._fee_base = fee_base
        self._fee_rate = fee_rate
        self._verbose = verbose

    def init(self, inputs, timestamp: int) -> SimpleTraderState:
        n = self._num_stocks if self._num_stocks is not None else inputs[0].shape[0]
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
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        N = state.num_stocks
        soft_positions = inputs[0].value()
        closes = inputs[1].value()
        adjusts = inputs[2].value()
        upper_limit = inputs[3].value()
        lower_limit = inputs[4].value()

        # Adjust shares for dividends (reinvesting all dividends).
        valid_adjusts = np.isfinite(adjusts) & (adjusts > 0)
        adjust_mask = valid_adjusts & (state.last_adjust > 0)
        state.shares[adjust_mask] *= adjusts[adjust_mask] / state.last_adjust[adjust_mask]
        state.last_adjust[valid_adjusts] = adjusts[valid_adjusts]

        # Update the last-valid-close carry-forward for stocks that
        # ticked this cycle; suspended stocks retain their previous
        # last-valid close.  Done BEFORE the rebalance because today's
        # close is now both the execution price and the mark-to-market
        # price - the same value is used to size, fill, and value the
        # post-trade position, so there is no intraday lookahead.
        close_tick = np.isfinite(closes)
        state.last_close[close_tick] = closes[close_tick]

        # Execute pending rebalance from the previous tick: signal
        # observed one tick ago, executed at today's close (MOC-style),
        # sized at today's close.  Trading on the closing-auction print
        # is the dominant venue for institutional systematic strategies
        # and is closer to VWAP than open execution.  The one-tick
        # delay keeps the signal information set strictly older than
        # the execution price (signal at close[t] -> execution at
        # close[t+1]).
        traded = False
        if state.pending_positions is not None:
            pending = state.pending_positions
            valid_exec = np.isfinite(closes) & (closes > 0)

            # Step 1: force-liquidate held positions in stocks with no
            # valid exec price today (suspended or delisted) at their
            # last valid close - the simulator assumes an idealised
            # exit even when no closing-auction trade is actually
            # possible.  Each forced sale is charged the same fee as a
            # normal trade.
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
            state._exec_price = closes
            trade_lots = state.trade_fn(state, pending)

            # Step 3: execute the net delta lots at today's close for
            # tradable stocks only.  Per-stock loop because of
            # sub-lot-remnant liquidation, per-trade fees, and the
            # per-side A-shares price-limit block (no buys at limit-up,
            # no sells at limit-down).
            for i in range(N):
                if not valid_exec[i]:
                    continue
                p = closes[i]

                # Get share counts from lot counts.
                trade_shares = trade_lots[i] * state.lot_size

                # Liquidate sub-lot remnants.
                if abs(state.shares[i] + trade_shares) < state.lot_size:
                    trade_shares = -state.shares[i]

                # Enforce A-shares-style price-limit rules.  At
                # limit-up the order book has no sellers (buys cannot
                # fill); at limit-down it has no buyers (sells cannot
                # fill).  NaN in either limit means "no constraint".
                if trade_shares > 0 and np.isfinite(upper_limit[i]) and p >= upper_limit[i]:
                    continue
                if trade_shares < 0 and np.isfinite(lower_limit[i]) and p <= lower_limit[i]:
                    continue

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

        # Output `(holdings_value, cash)`.  Total portfolio value = sum.
        held = (state.shares != 0) & np.isfinite(state.last_close)
        holdings_value = np.sum(state.shares[held] * state.last_close[held])

        # Bankruptcy check.  A leveraged or sum-zero portfolio (e.g. a
        # long-short construction with gross > 0 and net == 0) can
        # accumulate enough adverse PnL on one leg to push total NAV
        # to zero or below; in a real account the broker margin-calls
        # and force-liquidates everything.  Emulate that here by
        # zeroing cash, shares, and any pending signal whenever NAV is
        # non-positive at the tick boundary.  The wiped-out state is
        # absorbing: on subsequent ticks `current_value` stays at 0,
        # so all rebalances trade nothing and every subsequent output
        # is `(0, 0)`.  Downstream plots are responsible for rendering
        # this as a gap (NaN) rather than a literal `log(0) = -inf`.
        if not (state.cash + holdings_value > 0):
            state.shares[:] = 0.0
            state.cash = np.array(0.0)
            state.pending_positions = None
            holdings_value = 0.0

        output.write(np.array([holdings_value, state.cash], dtype=np.float64))
        return True
