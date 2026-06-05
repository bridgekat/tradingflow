"""Benchmark leaf operator - frictionless ideal portfolio replication.

Ported from ``tradingflow.operators.traders.benchmark``.  Replicates the
soft position weights exactly: no fees, no lot rounding, instant fills.
``init`` / ``compute`` bodies are verbatim; only the framework imports
and handle-based ``__init__`` are replaced with config scalars.

Inputs (array views, all shape ``(num_stocks,)``):
    0. soft positions
    1. close prices (execution + mark-to-market)
    2. adjusts
    3. upper price limit
    4. lower price limit
Output: array view of shape ``(2,)`` -> ``(holdings_value, cash)``.

build kwargs:
    num_stocks (int, optional; else read from inputs[0].shape at init)
    initial_cash (float)
    use_adjusts (bool, default True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class BenchmarkState:
    # Configuration.
    num_stocks: int
    initial_cash: float
    use_adjusts: bool

    # Portfolio state.
    cash: np.ndarray = field(default_factory=lambda: np.array(0.0))
    shares: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_adjust: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_close: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Rebalance signal observed on the previous tick, awaiting
    # execution at today's open.  `None` when there is no pending
    # signal.  Deferring execution by one tick keeps the signal's
    # information set (universe construction, factor values, ...)
    # strictly older than the execution price.
    pending_positions: np.ndarray | None = None


class Benchmark:
    """Frictionless benchmark that replicates soft position weights exactly.

    See the upstream docstring for the full description of the
    valuation, force-liquidation, and price-limit behaviour.
    """

    def __init__(
        self,
        *,
        num_stocks: int,
        initial_cash: float,
        use_adjusts: bool,
    ) -> None:
        self._num_stocks = num_stocks
        self._initial_cash = initial_cash
        self._use_adjusts = use_adjusts

    def init(self, inputs, timestamp: int) -> BenchmarkState:
        n = self._num_stocks if self._num_stocks is not None else inputs[0].shape[0]
        return BenchmarkState(
            num_stocks=n,
            initial_cash=self._initial_cash,
            use_adjusts=self._use_adjusts,
            cash=np.array(self._initial_cash),
            shares=np.zeros(n),
            last_adjust=np.ones(n),
            last_close=np.full(n, np.nan),
        )

    @staticmethod
    def compute(
        state: BenchmarkState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        soft_positions = inputs[0].value()
        closes = inputs[1].value()
        adjusts = inputs[2].value()
        upper_limit = inputs[3].value()
        lower_limit = inputs[4].value()

        # Adjust shares for dividends (reinvesting all dividends).
        if state.use_adjusts:
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
        # close[t+1]), and using today's close for both sizing and
        # execution removes any intraday open/close mismatch.
        if state.pending_positions is not None:
            pending = state.pending_positions
            valid_exec = np.isfinite(closes) & (closes > 0)

            # Step 1: force-liquidate held positions in stocks with no
            # valid exec price today (suspended or delisted) at their
            # last valid close - the simulator assumes an idealised
            # exit even when no open-market trade is actually possible.
            force_liq = (state.shares != 0) & ~valid_exec & np.isfinite(state.last_close)
            state.cash += np.sum(state.shares[force_liq] * state.last_close[force_liq])
            state.shares[force_liq] = 0.0

            # Step 2: compute portfolio value post force-liquidation
            # using today's close, then rebalance tradable stocks to
            # target via a single net trade at today's close.  (In a
            # frictionless benchmark this coincides with full liquidate
            # + re-enter; the delta form keeps the logic consistent
            # with `SimpleTrader`, which must use it to avoid doubling
            # fees.)
            held = (state.shares != 0) & np.isfinite(state.last_close)
            current_value = state.cash + np.sum(state.shares[held] * state.last_close[held])
            safe_closes = np.where(valid_exec, closes, 1.0)
            target_shares = pending * current_value / safe_closes
            trade_shares = np.where(valid_exec, target_shares - state.shares, 0.0)

            # Step 3: enforce A-shares-style price-limit rules.  At
            # limit-up the order book has no sellers (buys cannot fill);
            # at limit-down it has no buyers (sells cannot fill).  Block
            # the corresponding side of any limit-locked stock by
            # zeroing its trade_shares; NaN in either limit is treated
            # as "no constraint" (e.g. first tick before history exists,
            # or markets without a limit rule).
            block_buy = np.isfinite(upper_limit) & (closes >= upper_limit) & (trade_shares > 0)
            block_sell = np.isfinite(lower_limit) & (closes <= lower_limit) & (trade_shares < 0)
            trade_shares = np.where(block_buy | block_sell, 0.0, trade_shares)

            state.cash -= np.sum(trade_shares * safe_closes)
            state.shares += trade_shares

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


def build(**kwargs) -> Benchmark:
    return Benchmark(
        num_stocks=kwargs.get("num_stocks", None),
        initial_cash=kwargs.get("initial_cash", 1_000_000.0),
        use_adjusts=kwargs.get("use_adjusts", True),
    )
