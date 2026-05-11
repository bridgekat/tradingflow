"""Benchmark operator - frictionless ideal portfolio replication."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


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


class Benchmark(
    Operator[
        ArrayView[np.float64],  # soft positions (num_stocks,)
        ArrayView[np.float64],  # close prices (num_stocks,)
        ArrayView[np.float64],  # adjusts (num_stocks,)
        ArrayView[np.float64],  # upper price limit (num_stocks,)
        ArrayView[np.float64],  # lower price limit (num_stocks,)
        ArrayView[np.float64],  # output: (holdings_value, cash)
        BenchmarkState,
    ]
):
    """Frictionless benchmark that replicates soft position weights exactly.

    On every tick:

    1. Adjusts held shares for dividend reinvestment via forward
       adjustment factor changes.
    2. If a rebalance signal was received on the **previous** tick,
       first force-liquidates held positions in stocks with no valid
       exec price today (suspended or delisted) at their last valid
       close, then rebalances the remaining tradable stocks to their
       target allocations via a single net trade at today's close
       (MOC-style):
       `target_shares = pending_positions * current_value / close_price`,
       executed as fractional shares with no transaction fees and no
       lot rounding.  Trades that would violate today's daily price
       limit are blocked: buys are skipped in stocks whose close is at
       or above ``upper_limit`` (limit-up, no sellers), and sells are
       skipped in stocks whose close is at or below ``lower_limit``
       (limit-down, no buyers).  Suspended or delisted stocks hold
       zero shares post-rebalance.
    3. If the soft-positions input was updated this tick, stores it as
       the pending signal for execution on the **next** tick.  Trading
       is therefore deferred by one tick, so today's signal (derived
       from data observable at today's close) is executed at tomorrow's
       close - the upstream signal construction never has access to the
       execution-price information set.
    4. Outputs a 2-element array `(holdings_value, cash)` where
       `holdings_value` is positions valued at closing prices and
       `cash` is the cash balance.  Total portfolio value is their sum.

    Parameters
    ----------
    soft_positions
        Soft position weights, shape `(num_stocks,)`.
    close
        Stacked unadjusted close prices, shape `(num_stocks,)`.  Used
        both as the execution price (MOC at today's close) and as the
        mark-to-market price.
    adjusts
        Stacked forward adjustment factors, shape `(num_stocks,)`.
    upper_limit, lower_limit
        Today's daily price-limit thresholds, shape `(num_stocks,)`.
        On every tick the rebalance blocks **buys** in stocks whose
        close has reached or exceeded ``upper_limit`` and **sells** in
        stocks whose close has reached or fallen below ``lower_limit``
        - matching A-shares limit-locked behaviour where the order book
        is one-sided at the limit price.  `NaN` in either handle is
        treated as "no constraint" for that side.  Typically computed
        upstream via
        [`build_price_limits`][common.build_price_limits].
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

    **NaN prices (suspended / delisted stocks).**  A stock with `NaN`
    close on the current tick is handled as follows:

    - **Valuation (every tick)**: portfolio value uses the *last valid
      close* carried forward, so held shares of a suspended stock keep
      contributing at their most recent known price instead of dropping
      out.
    - **Rebalance**: stocks with a finite, positive current close are
      rebalanced via a single net trade (delta shares from current to
      target).  Stocks with non-finite or non-positive close cannot be
      traded at the close; if we held any, the simulator force-closes
      them at their last valid close (an idealisation that is not
      achievable in live trading) so they hold zero shares
      post-rebalance.  No fresh entry is ever made into a stock whose
      current close is invalid - capital earmarked for it remains in
      cash.
    """

    def __init__(
        self,
        soft_positions: Handle,
        close: Handle,
        adjusts: Handle,
        upper_limit: Handle,
        lower_limit: Handle,
        *,
        initial_cash: float,
        use_adjusts: bool,
    ) -> None:
        assert len(soft_positions.shape) == 1, "Soft positions input must have shape (num_stocks,)."
        assert len(close.shape) == 1, "Close input must have shape (num_stocks,)."
        assert len(upper_limit.shape) == 1, "Upper limit input must have shape (num_stocks,)."
        assert len(lower_limit.shape) == 1, "Lower limit input must have shape (num_stocks,)."
        assert (
            soft_positions.shape[0] == close.shape[0] == upper_limit.shape[0] == lower_limit.shape[0]
        ), "All input handles must agree on the number of stocks."

        self._num_stocks = soft_positions.shape[0]
        self._initial_cash = initial_cash
        self._use_adjusts = use_adjusts

        super().__init__(
            inputs=(soft_positions, close, adjusts, upper_limit, lower_limit),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(2,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> BenchmarkState:
        n = self._num_stocks
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
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
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
